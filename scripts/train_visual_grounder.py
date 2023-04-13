from ns_man.structs import *
from ns_man.utils.scene_graph import *
from ns_man.utils.load_dataset import get_sim_rgbd_scenes
from ns_man.perception.visual.visual_features import make_visual_embedder
from ns_man.language.word_embedding import make_word_embedder
from ns_man.grounders.visual import VisualGrounder

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import random
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR

# random.seed(22)


def make_visual_grounder_dataset(json_train_path: str,
                json_val_path: str,
                metadata_path: str,
                catalogue_path: str,
                synonyms_path: str,
                vfeats_chp_path: str,
                save: Maybe[str] = None
) -> Dict[str, List[Tuple[Tensor, Tensor]]]:

  # glove embeddings
  WE = make_word_embedder('glove_sm')

  # load attribute metadata
  metadata = json.load(open(metadata_path))
  catalogue = pd.read_table(catalogue_path)
  synonyms = json.load(open(synonyms_path))

  # load extracted visual features checkpoint
  vfeats_chp = torch.load(vfeats_chp_path)

  # loading scene graphs
  scenes_train = json.load(open(json_train_path))['scenes']
  scenes_val = json.load(open(json_val_path))['scenes']

  result = {}
  for (split, scene_graphs) in [('train', scenes_train), ('val', scenes_val)]:
    ds = []
    print(f'Doing {split}...')
    for scene_graph in tqdm(scene_graphs):
      n_objects = len(scene_graph['objects'])
      vfeats = vfeats_chp[split][scene_graph['image_index']]['visual_features']
      attr_labels = [a[1:] for a in vfeats_chp[split][scene_graph['image_index']]['attribute_labels']]
      obj_names = [a[0] for a in vfeats_chp[split][scene_graph['image_index']]['attribute_labels']]
      assert n_objects == vfeats.shape[0] == len(attr_labels)

      # sample positive queries and a negative pair of same concept (text-rank)
      q_pos, q_neg = [], [], 
      for i, labels in enumerate(attr_labels):
        has_syn = [] if not obj_names[i] in synonyms.keys() else [obj_names[i]]
        w_pos = random.choice(labels + has_syn)
        w_pos_syn = random.choice(synonyms[w_pos])
        for concept, value_list in metadata['types'].items():
          if value_list is None:
            continue
          if w_pos in value_list:
            choose_from = value_list[:]
            choose_from.remove(w_pos)
            w_neg = random.choice(choose_from)
            w_neg_syn = random.choice(synonyms[w_neg])
            break
        q_pos.append([w_pos_syn, w_pos])
        q_neg.append([w_neg_syn, w_neg])

      q_emb_pos = torch.stack([WE(q[0]).mean(0) for q in q_pos])
      q_emb_neg = torch.stack([WE(q[0]).mean(0) for q in q_neg]) 

      # make a permutation of vfeats so that we have negative v-q pairs (vis-rank)
      obj_ids = list(range(n_objects))
      sample_perms = [random.sample(obj_ids, n_objects) for _ in range(5000)]
      _okay = False
      for perm in sample_perms:
        _okay = True
        for i, n in enumerate(perm):
          qi = q_pos[i][1]
          if qi in attr_labels[n] + [obj_names[n]]:
            _okay = False
            break
        if _okay:
          break
      if not _okay:
        continue

      _sample = {'vis_emb': vfeats, 
                 'q_emb_pos': q_emb_pos,
                 'q_emb_neg': q_emb_neg,
                 'q_pos': [q[0] for q in q_pos],
                 'q_neg': [q[0] for q in q_neg],
                 'perm': perm,
                 'obj_names': obj_names
                }
      ds.append(_sample)

    result[split] = ds
  
  return result



def train_visual_grounder(train_checkpoint_path: str,
                           val_checkpoint_path: str,
                           n_epochs: int,
                           batch_size: int,
                           lr: float,
                           wd: float,
                           dropout: float,
                           device: str,
                           stop_patience: int,
                           x_size: int = 14,
                           qemb_size: int = 96,
                           jemb_size: int = 96,
                           save: Maybe[str] = None,
):
  print('Loading...')
  model = VisualGrounder(x_size, qemb_size, jemb_size, dropout).to(device)
  print(model)
  train_ds, val_ds = torch.load(train_checkpoint_path), torch.load(val_checkpoint_path)
  collator = model.make_collate_fn
  train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, 
    collate_fn=collator(device))
  val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, 
    collate_fn=collator(device))
  opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
  scheduler = CosineAnnealingLR(opt,
                              T_max = n_epochs, # Maximum number of iterations.
                              eta_min = 1e-5) # Minimum learning rate.
  max_eval = 1e2; patience = stop_patience
  for epoch in range(n_epochs):
    train_metrics = model.train_step(train_dl, opt, scheduler)
    eval_metrics = model.eval_step(val_dl)
    if eval_metrics['loss'] < max_eval:
      max_eval = eval_metrics['loss']
      patience = stop_patience
      if save is not None:
        torch.save(model.state_dict(), save)
    else:
      patience -= 1
      if not patience:
        break
    print(f"Epoch {epoch+1}/{n_epochs}: Train = {train_metrics}, Val = {eval_metrics}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=256)
    parser.add_argument('-e', '--n_epochs', help='how many epochs of training', type=int, default=10)
    parser.add_argument('-s', '--save', help='where to save best model weights', type=str, default=None)
    parser.add_argument('-wd', '--wd', help='weight decay to use for regularization', type=float, default=0.)
    parser.add_argument('-early', '--stop_patience', help='early stop patience (default no early stopping)', type=int, default=4)
    parser.add_argument('-lr', '--lr', help='learning rate to use in optimizer', type=float, default=1e-03)
    parser.add_argument('-dr', '--dropout', help='dropout for model', type=float, default=0.)
    parser.add_argument('-xs', '--x_size', help='number of spatial features per object', type=int, default=2048)
    parser.add_argument('-qs', '--qemb_size', help='query embedding size', type=int, default=96)
    parser.add_argument('-js', '--jemb_size', help='joint embedding size for model', type=int, default=2048)
    parser.add_argument('-tp', '--train_checkpoint_path', help='checkpoint path for train dataset', type=str, default=None)    
    parser.add_argument('-vp', '--val_checkpoint_path', help='checkpoint path for validation dataset', type=str, default=None)    
    kwargs = vars(parser.parse_args())
    train_visual_grounder(**kwargs)