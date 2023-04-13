from ns_man.structs import *
from ns_man.language.word_embedding import make_word_embedder
from ns_man.language.seq2seq import make_seq2seq_net, Seq2Seq
from ns_man.programs.tokenizer import *

from math import ceil
import json
import os

import torch
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split 
from torch.optim import AdamW, Adam, Optimizer 
from torch.optim.lr_scheduler import ReduceLROnPlateau

# reproducability
SEED = torch.manual_seed(9)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(9)


def make_q_vocab(questions: List[str]):
    all_tokens = set([])
    for question in questions:
        tokens = question.split()
        for t in tokens:
            all_tokens.add(t)
    q_vocab = {
        **SPECIAL_TOKENS,
        **{v: k + len(SPECIAL_TOKENS) for k, v in enumerate(sorted(all_tokens))},
    }
    return q_vocab
    #print(q_vocab)

def tokenize_questions(questions: List[str], 
                        word2id,
                        max_len: Maybe[int] = None) -> List[Tensor]:
    token_ids = []
    for question in questions:
        tokens = ['<START>'] + question.split() + ['<END>']
        token_ids.append(torch.tensor([word2id[t] for t in tokens]).long())
    if max_len is not None:
        token_ids = pad_sequence(
            token_ids + [torch.empty(max_len, dtype=longt)],
            padding_value=SPECIAL_TOKENS["<PAD>"],
            batch_first=True,
        )[:-1]
        assert token_ids.shape[0] == len(questions)
    return token_ids

def embed_questions_raw(questions: List[str], 
                        q_vocab: Dict[str, int],
                        max_len: Maybe[int] = None,
) -> List[Tensor]:
    # all_tokens = set([])
    # for question in questions:
    #     tokens = question.split()
    #     for t in tokens:
    #         all_tokens.add(t)
    # q_vocab = {
    #     **SPECIAL_TOKENS,
    #     **{v: k + len(SPECIAL_TOKENS) for k, v in enumerate(sorted(all_tokens))},
    # }
    #print(q_vocab)
    token_ids = tokenize_questions(questions, q_vocab, max_len)
    word2id = q_vocab
    id2word = {v:k for k,v in q_vocab.items()}
    return token_ids, list(q_vocab.keys()), word2id, id2word


def embed_questions_glove(questions: List[str],
                        max_len: Maybe[int] = None,
                        save_checkpoint: Maybe[str] = None,
                        remove_punct: bool = True,
                        ) -> Tensor:
    WE = make_word_embedder(remove_punct=remove_punct)
    questions_vectorized = list(map(WE, questions))
    if max_len is not None:
        embedding_size = questions_vectorized[0].shape[-1]
        questions_vectorized = pad_sequence(
            questions_vectorized + [torch.empty(max_len, embedding_size)],
            padding_value=SPECIAL_TOKENS["<PAD>"],
            batch_first=True,
        )[:-1]
        assert questions_vectorized.shape[0] == len(questions)
    if save_checkpoint is not None:
        torch.save(questions_vectorized, save_checkpoint)
    return questions_vectorized


def make_vectorized_seq2seq_dataset(questions, programs, q_families,
                            glove_embeddings: bool = False,
                            save_checkpoint: Maybe[str] = None,
                            save_vocabs: Maybe[str] = None,
                            save_gloves_checkpoint: Maybe[str] = None,
                            load_gloves_checkpoint: Maybe[str] = None
                        ):
    if glove_embeddings:
        word_vectors = embed_questions_glove(questions, None, save_gloves_checkpoint) if load_gloves_checkpoint is None else torch.load(load_gloves_checkpoint)['word_vectors'] 
        question_vocab, word2id, id2word = None, None, None
        prog_ids, prog_vocab = embed_programs(programs, max_len=None)
    else:
        q_vocab = make_q_vocab(questions)
        word_vectors, question_vocab, word2id, id2word = embed_questions_raw(questions, q_vocab, max_len=None)
        prog_ids, prog_vocab = embed_programs(programs, max_len=None)
    prog2id = {v:k for k,v in prog_vocab.items()}
    id2prog = prog_vocab                       
    dataset = {
        'program_vocab': prog_vocab,
        'question_vocab': question_vocab,
        'prog2id': prog2id, 'id2prog': id2prog,
        'word2id': word2id, 'id2word': id2word,
        'questions': word_vectors,
        'programs':  prog_ids,
        'question_family_indices': q_families
    }
    if save_vocabs is not None:
        with open(save_vocabs, 'w') as g:
            json.dump({'prog2id': prog2id, 'word2id': word2id}, g)
    if save_checkpoint is not None:
        torch.save(dataset, save_checkpoint)
    return dataset


# token and sequence level accuracy metrics for seq2seq decoding
# manually remove padded tokens
@torch.no_grad()
def accuracy_metrics(predictions: Tensor, truth: Tensor, ignore_idx: int) -> Tuple[int, ...]:
    num_sents_total = predictions.shape[0]
    num_items_total = predictions.shape[0] * predictions.shape[1]

    correct_items = torch.ones_like(predictions, device=predictions.device)
    correct_items[predictions != truth] = 0
    correct_items[truth == ignore_idx] = 1

    correct_sents = correct_items.prod(dim=1)
    num_correct_sents = correct_sents.sum().item()

    num_masked_items = len(truth[truth == ignore_idx])
    num_correct_items = correct_items.sum().item() - num_masked_items
    num_items_total -= num_masked_items

    return num_sents_total, num_correct_sents, num_items_total, num_correct_items


def train_step(model: Seq2Seq, 
               dl: DataLoader, 
               opt: Optimizer, 
               crit: nn.Module
               ):
    model.train()

    epoch_loss = 0.
    num_items, num_sents, num_correct_items, num_correct_sents = 0, 0, 0, 0 
    for batch_idx, (src, tgt) in enumerate(dl):
        out = model.forward(src, tgt[:, :-1]) # B x L x |V|
        tgt = tgt[:, 1:]
        #out = out.view(-1, out.shape[-1])
        #tgt = tgt[1:].view(-1)

        # backward
        opt.zero_grad()
        loss = crit(out.view(-1, out.shape[-1]), tgt.flatten()) # B*L x |V|
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        opt.step()

        # metrics
        epoch_loss += loss.item()
        metrics = accuracy_metrics(out.argmax(-1), tgt, model.pad_id) # B x L 
        num_sents += metrics[0]
        num_correct_sents += metrics[1]
        num_items += metrics[2]
        num_correct_items += metrics[3]

    epoch_loss /= len(dl)
    item_accuracy = num_correct_items / num_items 
    sent_accuracy = num_correct_sents / num_sents
    return epoch_loss, item_accuracy, sent_accuracy


@torch.no_grad()
def eval_step(model: Seq2Seq, 
              dl: DataLoader, 
              crit: nn.Module
            ):
    model.eval()

    epoch_loss = 0.
    num_items, num_sents, num_correct_items, num_correct_sents = 0, 0, 0, 0 
    for batch_idx, (src, tgt) in enumerate(dl):
        out = model.forward(src, tgt[:, :-1]) # B x L x |V|
        tgt = tgt[:, 1:]

        # metrics
        loss = crit(out.view(-1, out.shape[-1]), tgt.flatten()) # B*L x |V|
        epoch_loss += loss.item()
        metrics = accuracy_metrics(out.argmax(-1), tgt, model.pad_id) # B x L 
        num_sents += metrics[0]
        num_correct_sents += metrics[1]
        num_items += metrics[2]
        num_correct_items += metrics[3]

    epoch_loss /= len(dl)
    item_accuracy = num_correct_items / num_items 
    sent_accuracy = num_correct_sents / num_sents
    return epoch_loss, item_accuracy, sent_accuracy


def subsample_per_family(ds: List[Dict[str, Any]], n_samples_per_family: int):
    res, res_ids = [], []
    families = {}
    for i, sample in enumearte(ds):
        q_fam = sample['question_family_index']
        if q_fam not in families.keys():
            families[q_fam] = 1
            res.append(sample); res_ids.append(i)
        elif families[q_fam] < n_samples_per_family:
            families[q_fam] += 1
            res.append(sample); res_ids.append(i)
    return res, res_ids

                                  
def train_seq2seq(train_chp_path: str,
         dev_chp_path: str,
         model_config: Union[str, Dict[str, Any]],
         num_epochs: int,
         batch_size: int,
         lr: float,
         wd: float,
         adam_factor: float,
         stop_patience: int,
         device: str,
         replaced: bool = True,
         n_data: Maybe[int] = None,
         save_checkpoint: Maybe[str] = None,
         verbose: bool=True,
         test_gen: bool=False
        ):
    print(f'Loading data from {train_chp_path, dev_chp_path}..')
    print(f'Loading model config from {model_config}')
    model_config = json.load(open(model_config)) if type(model_config) == str else model_config
    
    train_ds = torch.load(train_chp_path)
    dev_ds = torch.load(dev_chp_path)
    
    if n_data is not None:
        n_samples_per_family = n_data // 60
        print(n_samples_per_family, n_data)
        # subsample n_data examples per unique question family
        train_ds_keep = []
        families = {}
        for i, q_fam in enumerate(train_ds['question_family_indices']):
            if q_fam not in families.keys():
                families[q_fam] = 1
                train_ds_keep.append([train_ds["questions"][i], train_ds["programs"][i]])
            elif families[q_fam] < n_samples_per_family:
                families[q_fam] += 1
                train_ds_keep.append([train_ds["questions"][i], train_ds["programs"][i]])
        train_ds = train_ds_keep; del train_ds_keep
    else:
        train_ds = list(zip(train_ds['questions'], train_ds['programs']))
    
    dev_ds = list(zip(dev_ds['questions'], dev_ds['programs']))
    test_ds = None

    if test_gen:
        # the dev_ds here is for testing, use random split for dev
        test_ds = dev_ds[:]
        total_size = len(train_ds)
        dev_size = ceil(.15 * total_size)
        train_ds, dev_ds = random_split(train_ds, [total_size - dev_size, dev_size])
                                  
    model = make_seq2seq_net(model_config).to(device)
    #model = Transformer(model_config).to(device)
    optim = Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optim, 
                                   factor=adam_factor, 
                                   patience=10,
                                   verbose=True
                                   )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=model.make_collate_fn(device, not replaced))
    dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=model.make_collate_fn(device, not replaced))

    crit = nn.CrossEntropyLoss(ignore_index=model.pad_id).to(device)

    losses, accus_item, accus_sent = [], [], []
    stop_patience = stop_patience if stop_patience is not None else num_epochs
    max_eval, patience = -1, stop_patience
    state_dict = {}
    print('Training..')
    for iteration in range(num_epochs):
        train_loss, train_accu_item, train_accu_prog = train_step(model, train_dl, optim, crit)
        dev_loss, dev_accu_item, dev_accu_prog = eval_step(model, dev_dl, crit)
        #test_loss, test_accu_item, test_accu_prog = eval_step(model, test_dl, crit)

        if verbose:
            print(f'Iteration {iteration+1}/{num_epochs}:\n \
                TRAIN: Loss={train_loss:.4f}\tAccuracy(token)={train_accu_item:.3f}\tAccuracy(prog)={train_accu_prog:.3f}\n \
                DEV: Loss={dev_loss:.4f}\tAccuracy(token)={dev_accu_item:.3f}\tAccuracy(prog)={dev_accu_prog:.3f}')#\
                #TEST: Loss={test_loss:.4f}\tAccuracy(token)={test_accu_item:.3f}\tAccuracy(prog)={test_accu_prog:.3f}\n')

        # early stopping
        if dev_accu_prog > max_eval:
            max_eval = dev_accu_prog
            patience = stop_patience
            if save_checkpoint is not None:
                torch.save(model.state_dict(), save_checkpoint)
        else:
            patience -=1 
            if not patience:
                break

    if test_ds is not None:
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=model.make_collate_fn(device, not replaced)) 
        test_loss, test_accu_item, test_accu_prog = eval_step(model, test_dl, crit)
        print(f'TEST: Loss={test_loss:.4f}\tAccuracy(token)={test_accu_item:.3f}\tAccuracy(prog)={test_accu_prog:.3f}')
    
    return max_eval


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rep', '--replaced', default=True, action='store_false', help='whether to use replaced or end2end method (default replaced)')
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=256)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=50)
    parser.add_argument('-f', '--model_config', help='path to model configuration', type=str, default='./configs/SIM/seq2seq_rnn.json')
    parser.add_argument('-s', '--save_checkpoint', help='where to save best model weights', type=str, default=None)
    parser.add_argument('-wd', '--wd', help='weight decay to use for regularization', type=float, default=0.)
    parser.add_argument('-a', '--adam_factor', help='factor for Adam scheduling', type=float, default=0.)
    parser.add_argument('-early', '--stop_patience', help='early stop patience (default no early stopping)', type=int, default=4)
    parser.add_argument('-lr', '--lr', help='learning rate to use in optimizer', type=float, default=3e-04)
    parser.add_argument('-tp', '--train_chp_path', help='path to train split checkpoint', default=None)
    parser.add_argument('-vp', '--dev_chp_path', help='path to val split checkpoint', default=None)
    parser.add_argument('-n', '--n_data', help='number of training data to subsample (None for full dataset)', type=int, default=None)
    parser.add_argument('-ver', '--verbose', help='print training progress', type=bool, default=True)
    parser.add_argument('-gen', '--test_gen', default=False, action='store_true', help='training for generalization to unseen vocabulary experiment?')
    
    kwargs = vars(parser.parse_args())

    # first create checkpoint datasets if not already done
    if kwargs["train_chp_path"] is None:
      print('Making checkpoint datasets...')
      ds_train_path, ds_val_path = "./data/questions/SynHOTS-VQA_proc_train.json", "./data/questions/SynHOTS-VQA_proc_val.json"
      vocab_path, vocab_end2end_path = './checkpoints/Seq2seq/vocabs.json', './checkpoints/Seq2seq/vocabs_end2end.json'
      chp_train, chp_train_end2end = "./checkpoints/Seq2seq/ds_train_vectorized.p", "./checkpoints/Seq2seq/ds_train_end2end_vectorized.p"
      chp_val, chp_val_end2end =  "./checkpoints/Seq2seq/ds_val_vectorized.p", "./checkpoints/Seq2seq/ds_val_end2end_vectorized.p"
      load_glove_train, load_glove_val = "./checkpoints/Tag/ds_train_vectorized.p", "./checkpoints/Tag/ds_val_vectorized.p"
      if kwargs['test_gen']:
        print('Running for test-gen splits...')
        ds_train_path, ds_val_path = "./data/questions/SynHOTS-VQA_proc_TestGen_train.json", "./data/questions/SynHOTS-VQA_proc_TestGen_test.json"
        vocab_path, vocab_end2end_path = './checkpoints/TestGen/vocabs.json', './checkpoints/TestGen/vocabs_end2end.json'
        chp_train, chp_train_end2end = "./checkpoints/TestGen/ds_seq2seq_train_vectorized.p", "./checkpoints/TestGen/ds_seq2seq_train_end2end_vectorized.p"
        chp_val, chp_val_end2end =  "./checkpoints/TestGen/ds_seq2seq_test_vectorized.p", "./checkpoints/TestGen/ds_seq2seq_test_end2end_vectorized.p"
        load_glove_train, load_glove_val = "./checkpoints/TestGen/ds_tags_train_vectorized.p", "./checkpoints/TestGen/ds_tags_test_vectorized.p"
      
      ds_train = json.load(open(ds_train_path))
      ds_val = json.load(open(ds_val_path))
      
      queries_replaced = [x["seq2seq_input_single"] for x in ds_train]
      queries_raw = [x["tagger_input"] for x in ds_train]
      programs_replaced = [x["seq2seq_output_single"] for x in ds_train]
      programs_raw =  [x["groundtruth_program"] for x in ds_train]
      q_families = [x["question_family_index"] for x in ds_train]
      _ = make_vectorized_seq2seq_dataset(queries_replaced, programs_replaced, q_families, False, 
                                    save_vocabs=vocab_path,
                                    save_checkpoint=chp_train)
      _ = make_vectorized_seq2seq_dataset(queries_raw, programs_raw, q_families, True, 
                                    save_vocabs= vocab_end2end_path,
                                    save_checkpoint=chp_train_end2end,
                                    load_gloves_checkpoint=load_glove_train)

      queries_replaced = [x["seq2seq_input_single"] for x in ds_val]
      queries_raw = [x["tagger_input"] for x in ds_val]
      programs_replaced = [x["seq2seq_output_single"] for x in ds_val]
      programs_raw =  [x["groundtruth_program"] for x in ds_val]
      q_families = [x["question_family_index"] for x in ds_val]
      _ = make_vectorized_seq2seq_dataset(queries_replaced, programs_replaced, q_families, False, 
                                    save_checkpoint=chp_val)
      _ = make_vectorized_seq2seq_dataset(queries_raw, programs_raw, q_families, True, 
                                    save_checkpoint=chp_val_end2end,
                                    load_gloves_checkpoint=load_glove_val)

      del ds_val, ds_train, queries_replaced, queries_raw, programs_replaced, programs_raw
      print(kwargs['replaced'])
      kwargs["train_chp_path"] = chp_train if kwargs['replaced'] else chp_train_end2end
      kwargs["dev_chp_path"] = chp_val if kwargs['replaced'] else chp_val_end2end

    train_seq2seq(**kwargs)