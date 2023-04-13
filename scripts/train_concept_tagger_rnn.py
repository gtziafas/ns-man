from ns_man.structs import *
from ns_man.language.word_embedding import make_word_embedder
from ns_man.language.concept_tagger import make_concept_tagger_rnn, TAGSET

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Optimizer, Adam, AdamW
import json


# vectorize inputs-outputs
def make_vector_tagging_dataset(queries: Maybe[List[str]] = None, 
                                 tags: Maybe[List[str]] = None, 
                                 tagset: Maybe[List[str]] = None,
                                 load_checkpoint: Maybe[str] = None,
                                 save_checkpoint: Maybe[str] = None):

  if load_checkpoint is None:
    WE = make_word_embedder()
    #word_vectors = WE([q.strip('?').replace('-',' ') for q in queries]) # list of arrays
    word_vectors = list(map(WE, queries))
    #word_vectors = [torch.tensor(v) for v in word_vectors]

    # define tag vocabulary and tokenizer
    tag_vocab = {k: v for k,v in enumerate(tagset)}
    tag_vocab_inv = {v : k for k, v in tag_vocab.items()}
    
    # tokenize and convert to one-hot
    tag_ids = [torch.tensor([tag_vocab_inv[t] for t in ts.split()], dtype=longt) for ts in tags]
    tag_vectors = [torch.eye(len(tag_vocab))[t] for t in tag_ids]
    
    assert [wv.shape[0] == tv.shape[0] for wv,tv in zip(word_vectors, tag_vectors)]

    dataset = {'word_vectors': word_vectors,
               'tag_vectors': tag_vectors,
               'tag_vocab': tag_vocab,
               'tag_vocab_inv': tag_vocab_inv,
               'tagset': tagset
              }
    if save_checkpoint is not None:
      torch.save(dataset, save_checkpoint)

  else:
    dataset = torch.load(load_checkpoint)

  return dataset 


# hack from-scratch embeddings inside the checkpoint to not compute GloVes again --
# TODO: replace it in ``make_vector_tagging_dataset`` fn above
def add_from_scratch_embeddings(ds_train):
  # make input vocab for from-scratch embeddings
  vocab = set([])
  for sample in ds_train["data"]:
      q = sample["tagger_input"]
      tokens = q.split()
      for t in tokens:
          vocab.add(t)
  print(f'Vocab size = {len(vocab)}')
  word2id = {v:k for k,v in enumerate(sorted(list(vocab)))}
  id2word = {k:v for k,v in enumerate(sorted(list(vocab)))}

  def tokenize_dataset(ds):
      vectors = []
      for sample in ds:
          q = sample["tagger_input"]
          tokens = q.split()
          vectors.append(torch.as_tensor([word2id[t] for t in tokens], dtype=longt))
      return vectors

  # train_chp = "./checkpoints/Tag/SIM_tagging_train_vectorized.p"; dev_chp = "./checkpoints/Tag/SIM_tagging_dev_vectorized.p"
  # train_ds = torch.load(train_chp)
  # dev_ds = torch.load(dev_chp)
  # train_ds["word_ids"] = tokenize_dataset(ds_train["data"])
  # train_ds["word_vocab"] = word2id
  # train_ds["word_vocab_inv"] = id2word
  # train_ds["wordset"] = vocab
  # dev_ds["word_ids"] = tokenize_dataset(ds_dev["data"])
  # dev_ds["word_vocab"] = word2id
  # dev_ds["word_vocab_inv"] = id2word
  # dev_ds["wordset"] = vocab
  # torch.save(train_ds, train_chp)
  # torch.save(dev_ds, dev_chp)
  # del train_ds, dev_ds


# train-test loop
def train_concept_tagger(checkpoint_train, checkpoint_val, 
                      n_data=None,
                      input_dim=96,
                      hidden_dim=96,
                      n_layers=1,
                      num_epochs=50, 
                      batch_size=256, 
                      lr=1e-4, 
                      wd=0., 
                      adam_factor=0.1, 
                      stop_patience=5,
                      device="cuda",
                      model_config=None,
                      pretrain=True,
                      save=False,
                      test_gen=False
):
    print(f'Loading data from {checkpoint_train, checkpoint_val}...')
    train_ds = torch.load(checkpoint_train)
    k = 'word_vectors' if pretrain else 'word_ids'
    word_vocab_size = len(train_ds["wordset"]) if not pretrain else None
    if n_data is not None:
        keep_indices = random.sample(range(len(train_ds['word_vectors'])), n_data)
        train_ds[k] = [x for i,x in enumerate(train_ds[k]) if i in keep_indices]
        train_ds['tag_vectors'] = [x for i,x in enumerate(train_ds['tag_vectors']) if i in keep_indices]
        
    dev_ds = torch.load(checkpoint_val)
    tag_vocab = train_ds['tag_vocab']
    tagset = train_ds['tagset']
    train_ds = list(zip(train_ds[k], train_ds['tag_vectors']))
    dev_ds = list(zip(dev_ds[k], dev_ds['tag_vectors']))
    
    model = make_concept_tagger_rnn(model_config).to(device)
    crit = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-1).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(opt, 
                                   factor=adam_factor, 
                                   patience=10,
                                   verbose=True
                                   )

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, collate_fn=model.make_collate_fn(device, pretrain))
    dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=batch_size, collate_fn=model.make_collate_fn(device, pretrain))

    print('Training for {} epochs'.format(num_epochs))
    max_eval = -1 
    patience = stop_patience
    for epoch in range(num_epochs):
        train_loss = 0. 
        #num_sents, num_correct_sents, num_words, num_correct_words = 0,0,0,0
        model.train() 
        all_preds, all_ys = [], []
        for batch_idx, (x, y) in enumerate(train_dl):
            preds, _ = model(x)
            opt.zero_grad()
            loss = crit(preds.view(-1, len(tagset)), y.argmax(-1).flatten())
            loss.backward()
            opt.step()
            
            train_loss += loss.item()
            all_preds.extend(list(preds.detach().cpu()))
            all_ys.extend(list(y.detach().cpu()))

        train_loss /= len(train_dl)
        assert len(all_preds) == len(all_ys) == len(train_dl.dataset), ('train', len(all_preds),len(all_ys), len(train_dl.dataset))
        with torch.no_grad():
            all_preds = pad_sequence(all_preds, batch_first=True, padding_value=-1)
            all_ys = pad_sequence(all_ys, batch_first=True, padding_value=-1)
        train_metrics = model.compute_metrics_2(all_preds, all_ys)

        # testing 
        test_loss = 0. 
        #num_sents, num_correct_sents, num_words, num_correct_words = 0,0,0,0
        all_preds, all_ys = [], []
        model.eval()
        for x, y in dev_dl:
            with torch.no_grad():
                preds, _ = model(x)
                loss = crit(preds.view(-1, len(tagset)), y.argmax(-1).flatten())
            
            test_loss += loss.item()
            all_preds.extend(list(preds.detach().cpu()))
            all_ys.extend(list(y.detach().cpu()))

        test_loss /= len(dev_dl)
        assert len(all_preds) == len(all_ys) == len(dev_dl.dataset), ('dev', len(all_preds),len(all_ys), len(train_dl.dataset))
        with torch.no_grad():
            all_preds = pad_sequence(all_preds, batch_first=True, padding_value=-1)
            all_ys = pad_sequence(all_ys, batch_first=True, padding_value=-1)
        test_metrics = model.compute_metrics_2(all_preds, all_ys)

        # early stopping
        if test_metrics["f1"] > max_eval:
            max_eval = test_metrics["f1"]
            patience = stop_patience
            if save:
                torch.save(model.state_dict(), f"./checkpoints/{'Tag' if not test_gen else 'TestGen'}/tagger_rnn_weights.p")
        else:
            patience -=1 
            if not patience:
                break

        print('Epoch={}, train: Loss={:.5f}, Acc={:2.2f}, Precision={:2.2f}, Recall={:2.2f}, F1={:2.2f} \n\t test: Loss={:.5f}, Acc={:2.2f}, Precision={:2.2f}, Recall={:2.2f}, F1={:2.2f}'.format(epoch+1, 
          train_loss, 100*train_metrics["acc"], 100*train_metrics["p"], 100*train_metrics["r"], 100*train_metrics["f1"],
          test_loss, 100*test_metrics["acc"], 100*test_metrics["p"], 100*test_metrics["r"], 100*test_metrics["f1"]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=256)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=50)
    parser.add_argument('-f', '--model_config', help='path to model configuration', type=str, default=None)
    parser.add_argument('-s', '--save', help='where to save best model weights', type=bool, default=False)
    parser.add_argument('-wd', '--wd', help='weight decay to use for regularization', type=float, default=0.)
    parser.add_argument('-a', '--adam_factor', help='factor for Adam scheduling', type=float, default=0.)
    parser.add_argument('-early', '--stop_patience', help='early stop patience (default no early stopping)', type=int, default=2)
    parser.add_argument('-lr', '--lr', help='learning rate to use in optimizer', type=float, default=3e-04)
    parser.add_argument('-tp', '--checkpoint_train', help='path to train split of VQA dataset', default=None)
    parser.add_argument('-vp', '--checkpoint_val', help='path to val split of VQA dataset', default=None)
    parser.add_argument('-p', '--pretrain', help='whether to use pretrain word embeddings (must be consinstent with config)', type=bool, default=True)
    parser.add_argument('-n', '--n_data', help='number of training data to subsample (None for full dataset)', type=int, default=None)
    parser.add_argument('-gen', '--test_gen', default=False, action='store_true', help='training for generalization to unseen vocabulary experiment?')
    
    kwargs = vars(parser.parse_args())
    
    # first create checkpoint datasets if not already done
    if kwargs["checkpoint_train"] is None:
      print('Making checpoint datasets...')
      ds_train_path, ds_val_path = "./data/questions/SynHOTS-VQA_proc_train.json", "./data/questions/SynHOTS-VQA_proc_val.json"
      chp_train, chp_val = "./checkpoints/Tag/ds_train_vectorized.p", "./checkpoints/Tag/ds_val_vectorized.p"
      if kwargs['test_gen']:
        print('Running for test-gen splits...')
        ds_train_path, ds_val_path = "./data/questions/SynHOTS-VQA_proc_TestGen_train.json", "./data/questions/SynHOTS-VQA_proc_TestGen_test.json"
        chp_train, chp_val = "./checkpoints/TestGen/ds_tags_train_vectorized.p", "./checkpoints/TestGen/ds_tags_test_vectorized.p"
      
      ds_train = json.load(open(ds_train_path))
      ds_val = json.load(open(ds_val_path))

      queries = [x["tagger_input"] for x in ds_train]
      tags = [x["tagger_output"] for x in ds_train]
      _ = make_vector_tagging_dataset(queries, tags, TAGSET, load_checkpoint=None, save_checkpoint=chp_train)
      queries = [x["tagger_input"] for x in ds_val]
      tags = [x["tagger_output"] for x in ds_val]
      _ = make_vector_tagging_dataset(queries, tags, TAGSET, load_checkpoint=None, save_checkpoint=chp_val)
      del queries, tags, ds_train, ds_val
      kwargs["checkpoint_train"] = chp_train
      kwargs["checkpoint_val"] = chp_val

    train_concept_tagger(**kwargs)