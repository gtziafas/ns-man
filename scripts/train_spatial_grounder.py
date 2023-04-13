from ns_man.structs import *
from ns_man.utils.scene_graph import *
from ns_man.utils.load_dataset import get_sim_rgbd_scenes
from ns_man.perception.visual.visual_features import make_visual_embedder
from ns_man.language.word_embedding import make_word_embedder
from ns_man.grounders.spatial import RelationGrounder, BCEWithLogitsIgnore, HRelationGrounder, MSELossIgnore

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from tqdm import tqdm
import random
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR

random.seed(22)


def equalize_labels(ds):
  random.shuffle(ds)
  synonyms = REL_SYNONYMS
  keep = []
  for k in synonyms.keys():
    n_pos = sum([x['label'].item() for x in ds if x['q'] in synonyms[k]])
    _cnt = 0
    for sample in ds:
      q = sample['q']
      if q in synonyms[k]:
        if sample['label'].item():
          keep.append(sample)
        else:
          if _cnt < n_pos:
            keep.append(sample)
            _cnt += 1
  return keep


def balance_spatial_dataset(ds, synonyms: Dict[str, List[str]], n_samples_per_q: int = 5000):
  #synonyms = {k:v for k,v in synonyms.items() if k.upper() in RELATIONS}
  synonyms = REL_SYNONYMS
  # remove main diagonals
  neg = [x for x in ds if x['label'].item()==0]
  pos = [x for x in ds if x['label'].item()==1]
  n_neg, n_pos = len(neg), len(pos)
  # equalize 0-1 labels
  neg_filtered = random.sample(neg, n_pos)
  # equalize accross queries
  cnt = {k.lower(): 0 for k in RELATIONS}
  keep0 = []
  for sample in neg_filtered:
    q = sample['q']
    for rel in cnt.keys():
      if q in synonyms[rel] and cnt[rel] < n_samples_per_q:
        keep0.append(sample)
        cnt[rel] += 1
  cnt = {k.lower(): 0 for k in RELATIONS}
  keep1 = []
  for sample in pos:
    q = sample['q']
    for rel in cnt.keys():
      if q in synonyms[rel] and cnt[rel] < n_samples_per_q:
        keep1.append(sample)
        cnt[rel] += 1
  # shuffle
  ds_balanced = keep0 + keep1
  random.shuffle(ds_balanced)
  return ds_balanced



def make_spatial_grounder_dataset_features(json_train_path: str,
                json_val_path: str,
                H: int = 480,
                W: int = 640,
                save: Maybe[str] = None
) -> Dict[str, List[Tuple[Tensor, Tensor]]]:

  # load GloVe embedding
  WE = make_word_embedder('glove_sm')

  #synonyms = json.load(open(synonyms_path))
  synonyms = REL_SYNONYMS
  all_rel_words= sum(REL_SYNONYMS.values(), [])
  all_rel_embs = {q: WE(q).mean(0) for q in all_rel_words}

  # loading scene graphs
  scenes_train = json.load(open(json_train_path))['scenes']
  scenes_val = json.load(open(json_val_path))['scenes']

  result = {}
  dividend = torch.as_tensor([W, H, W, H, W, H, W, H], dtype=torch.float)
  for (split, scene_graphs) in [('train', scenes_train), ('val', scenes_val)]:
    ds = []
    print(f'Doing {split}...')
    for scene_graph in tqdm(scene_graphs):
      num_objects = len(scene_graph['objects'])
      rel_features = torch.from_numpy(vectorize_edges(scene_graph, skip_hrel=True)) # N x N x R
      x_rel = rearrange(rel_features, 'n m r -> (n m) r')

      for pair in x_rel:
        assert pair.shape[0] == 9
        for rel_idx, label in enumerate(pair):
          choices = synonyms[RELATIONS[rel_idx].lower()]
          query = random.choice(choices)  
          query_emb = all_rel_embs[query]
          ds.append({'x_pair': pair,
                      'q_emb': query_emb,
                      'q': query,
                      'label': label})
    result[split] = ds

  return result


def make_spatial_grounder_dataset_seq(json_train_path: str,
                json_val_path: str,
                synonyms_path: str,
                H: int = 480,
                W: int = 640,
                save: Maybe[str] = None
) -> Dict[str, List[Tuple[Tensor, Tensor]]]:

  # load GloVe embedding
  WE = make_word_embedder('glove_sm')

  #synonyms = json.load(open(synonyms_path))
  synonyms = REL_SYNONYMS

  # loading scene graphs
  scenes_train = json.load(open(json_train_path))['scenes']
  scenes_val = json.load(open(json_val_path))['scenes']

  result = {}
  dividend = torch.as_tensor([W, H, W, H, W, H, W, H], dtype=torch.float)
  for (split, scene_graphs) in [('train', scenes_train), ('val', scenes_val)]:
    ds = []
    print(f'Doing {split}...')
    for scene_graph in tqdm(scene_graphs):
      num_objects = len(scene_graph['objects'])
      # N x 11 spatial features
      spt_feats_2d_c = [torch.as_tensor(o['RGB_center'], dtype=torch.float) / dividend[:2] for o in scene_graph['objects']]
      spt_feats_2d_r = [torch.as_tensor(o['RGB_rectangle'], dtype=torch.float) / dividend for o in scene_graph['objects']]
      # spt_feats_3d = [torch.as_tensor(normalize_coords(o['position_3d']), dtype=torch.float) for o in scene_graph['objects']]
      spt_feats_size = [torch.as_tensor([o['size'] / SIZE_NORM]) for o in scene_graph['objects']]
      # spt = torch.stack([torch.cat([x1,x2,x3,x4], dim=0) for x1,x2,x3,x4 in zip(spt_feats_2d_c, spt_feats_2d_r, spt_feats_3d, spt_feats_size)])
      spt = torch.stack([torch.cat([x1,x2,x3], dim=0) for x1,x2,x3 in zip(spt_feats_2d_c, spt_feats_2d_r, spt_feats_size)])
      pair_dists = torch.from_numpy(compute_pairwise_dist(scene_graph['objects'])).float()
      
      rel_labels = torch.from_numpy(vectorize_edges(scene_graph, skip_hrel=True)) # N x N x R
      #rel_labels = torch.from_numpy(rel_labels).long().contiguous().reshape(num_objects**2, -1).transpose(0,1)
      rel_labels = rearrange(rel_labels, "n m r -> r n m")

      for rel_idx, rel_label in enumerate(list(rel_labels)):
        choices = synonyms[RELATIONS[rel_idx].lower()]
        query = random.choice(choices)  
        query_emb = WE(query).mean(0) # De
        ds.append({'x_emb': spt, 'x_dist': pair_dists, 'q_emb':query_emb, 'q': query, 'label': rel_label})
        
    result[split] = ds

  return result


def make_hrel_dataset(json_train_path: str,
                json_val_path: str
) -> Dict[str, List[Tuple[Tensor, Tensor]]]:
  # load GloVe embedding
  WE = make_word_embedder('glove_sm')

  synonyms = HREL_SYNONYMS
  all_hrel_words= sum(HREL_SYNONYMS.values(), [])
  all_hrel_embs = {q: WE(q).mean(0) for q in all_hrel_words}

  # loading scene graphs
  scenes_train = json.load(open(json_train_path))['scenes']
  scenes_val = json.load(open(json_val_path))['scenes']

  result = {}
  for (split, scene_graphs) in [('train', scenes_train), ('val', scenes_val)]:
    ds = []
    print(f'Doing {split}...')
    for scene_graph in tqdm(scene_graphs):
      num_objects = len(scene_graph['objects'])
      pair_dists = torch.from_numpy(compute_pairwise_dist(scene_graph['objects'])).float()
      pair_dist_tile = pair_dists.unsqueeze(2).repeat(1, 1, num_objects)
      delta_x = pair_dist_tile - pair_dist_tile.transpose(1, 2) # N x N x N
      closer = (delta_x < 0)
      further = (delta_x > 0)
      closer = closer.flatten()
      further = further.flatten()
      inputs = delta_x.flatten() # N**3
    
      for k in HRELATIONS:
        q = random.choice(synonyms[k.lower()])
        q_emb = all_hrel_embs[q]
        q_emb_tile = q_emb.unsqueeze(0).repeat(num_objects**3, 1) # N**3 x De
        ds.extend([{'x': x, 'q': q, 'q_emb': q_emb, 'label': y}
          for x,q_emb,y in zip(inputs, q_emb_tile, eval(k.lower()))])
        ds.extend([{'x': x, 'q': q, 'q_emb': q_emb, 'label': y}
          for x,q_emb,y in zip(inputs, q_emb_tile, eval(k.lower()))])

      result[split] = ds

  return result

def make_spatial_grounder_dataset(json_train_path: str,
                json_val_path: str,
                synonyms_path: str,
                H: int = 480,
                W: int = 640,
                save: Maybe[str] = None
) -> Dict[str, List[Tuple[Tensor, Tensor]]]:

  # load GloVe embedding
  WE = make_word_embedder('glove_sm')

  #synonyms = json.load(open(synonyms_path))
  synonyms = REL_SYNONYMS

  # loading scene graphs
  scenes_train = json.load(open(json_train_path))['scenes']
  scenes_val = json.load(open(json_val_path))['scenes']

  result = {}
  dividend = torch.as_tensor([W, H, W, H, W, H, W, H], dtype=torch.float)
  for (split, scene_graphs) in [('train', scenes_train), ('val', scenes_val)]:
    ds = []
    print(f'Doing {split}...')
    for scene_graph in tqdm(scene_graphs):
      num_objects = len(scene_graph['objects'])
      # N x 11 spatial features
      spt_feats_2d_c = [torch.as_tensor(o['RGB_center'], dtype=torch.float) / dividend[:2] for o in scene_graph['objects']]
      spt_feats_2d_r = [torch.as_tensor(o['RGB_rectangle'], dtype=torch.float) / dividend for o in scene_graph['objects']]
      # spt_feats_3d = [torch.as_tensor(normalize_coords(o['position_3d']), dtype=torch.float) for o in scene_graph['objects']]
      spt_feats_size = [torch.as_tensor([o['size'] / SIZE_NORM]) for o in scene_graph['objects']]
      # spt = torch.stack([torch.cat([x1,x2,x3,x4], dim=0) for x1,x2,x3,x4 in zip(spt_feats_2d_c, spt_feats_2d_r, spt_feats_3d, spt_feats_size)])
      spt = torch.stack([torch.cat([x1,x2,x3], dim=0) for x1,x2,x3 in zip(spt_feats_2d_c, spt_feats_2d_r, spt_feats_size)])
      # N x N x 23 spatial-pair features
      x_tile = spt.unsqueeze(1).repeat(1, num_objects, 1)
      pair_dists = torch.from_numpy(compute_pairwise_dist(scene_graph['objects'])).float()
      x_pair = torch.cat([x_tile, x_tile.transpose(0,1), pair_dists.unsqueeze(2)], dim=2)  
      #x_pair = torch.cat([x_tile - x_tile.transpose(0,1), pair_dists.unsqueeze(2)], dim=2)  
      #x_pair = x_pair.view(num_objects**2, -1) # N^2 x 23
      x_pair = rearrange(x_pair, "n m d -> (n m) d")

      rel_labels = torch.from_numpy(vectorize_edges(scene_graph, skip_hrel=True)) # N x N x R
      #rel_labels = torch.from_numpy(rel_labels).long().contiguous().reshape(num_objects**2, -1).transpose(0,1)
      rel_labels = rearrange(rel_labels, "n m r -> r (n m)")

      for rel_idx, rel_label in enumerate(list(rel_labels)):
        choices = synonyms[RELATIONS[rel_idx].lower()]
        query = random.choice(choices)  
        query_emb = WE(query).mean(0) # De
        query_emb_tile = query_emb.unsqueeze(0).repeat(num_objects**2, 1) # N^2 x De
        assert query_emb_tile.shape[0] == x_pair.shape[0] == rel_label.shape[0]
        zipped = list(zip(list(x_pair), list(query_emb_tile), list(rel_label.unsqueeze(1))))
        _to_add = [{'x_pair': x, 'q_emb': qe, 'q': query, 'label': l.long()} for x,qe,l in zipped]
        for dp in _to_add:
          if dp['x_pair'].sum().item() > 0:
            ds.append(dp)
    result[split] = ds
  return result


def train_spatial_grounder(train_checkpoint_path: str,
                           val_checkpoint_path: str,
                           n_epochs: int,
                           batch_size: int,
                           lr: float,
                           wd: float,
                           dropout: float,
                           device: str,
                           stop_patience: int,
                           save: Maybe[str] = None,
                           seq: bool = False,
                           hrel: bool = False
):
  print('Loading...')
  model = RelationGrounder().to(device) if not hrel else HRelationGrounder().to(device)
  print(model)
  assert not hrel or not seq, 'Cannot run HRelationGrounder in seq-mode'
  train_ds, val_ds = torch.load(train_checkpoint_path), torch.load(val_checkpoint_path)
  collator = model.make_collate_fn if not seq else model.make_collate_fn_seq
  train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, 
    collate_fn=collator(device))
  val_dl = DataLoader(val_ds, shuffle=False, batch_size=batch_size, 
    collate_fn=collator(device))
  opt = Adam(model.parameters(), lr=lr, weight_decay=wd)
  # opt = SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
  scheduler = CosineAnnealingLR(opt,
                              T_max = n_epochs, # Maximum number of iterations.
                              eta_min = 1e-5) # Minimum learning rate.
  crit = nn.BCELoss().to(device) if not seq else MSELossIgnore(ignore_index=model.ignore_idx)
  max_eval = -1; patience = stop_patience
  train_fn = model.train_step if not seq else model.train_step_seq
  eval_fn = model.eval_step if not seq else model.eval_step_seq
  print('Training...')
  for epoch in range(n_epochs):
    train_metrics = train_fn(train_dl, crit, opt, scheduler)
    eval_metrics = eval_fn(val_dl, crit)
    if eval_metrics['f1'] > max_eval:
      max_eval = eval_metrics['f1']
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
    parser.add_argument('-tp', '--train_checkpoint_path', help='checkpoint path for train dataset', type=str, default=None)    
    parser.add_argument('-vp', '--val_checkpoint_path', help='checkpoint path for validation dataset', type=str, default=None)    
    parser.add_argument('-seq', '--seq', default=False, action='store_true', help='sequenced version of training?')
    parser.add_argument('-hrel', '--hrel', default=False, action='store_true', help='train hyper-relation grounder?')
    kwargs = vars(parser.parse_args())
    train_spatial_grounder(**kwargs)