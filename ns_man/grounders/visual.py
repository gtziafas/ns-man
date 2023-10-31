from ns_man.structs import *

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR


class MaxMarginLoss(nn.Module):
  def __init__(self, margin: float, vis_rank_weight: float, text_rank_weight: float):
    super().__init__()
    self.margin = margin
    self.vis_rank = vis_rank_weight > 0
    self.text_rank = text_rank_weight > 0 
    self.vis_rank_weight = vis_rank_weight
    self.text_rank_weight = text_rank_weight

  def forward(self, cossim: Tensor) -> Tensor:
    batch_size = cossim.shape[0]
    B = 0

    if self.vis_rank and not self.text_rank:
      B = batch_size // 2
      paired = cossim[:B]
      unpaired = cossim[B:]
      vis_rank_loss = self.vis_rank_weight * torch.clamp(self.margin + unpaired - paired, min=0)
      text_rank_loss = 0.
    
    elif not self.vis_rank and self.text_rank:
      B = batch_size // 2
      paired = cossim[:B]
      unpaired = cossim[B:]
      text_rank_loss = self.text_rank_weight * torch.clamp(self.margin + unpaired - paired, min=0)
      vis_rank_loss = 0.
    
    elif self.vis_rank and self.text_rank:
      B = batch_size // 3
      paired = cossim[:B]
      text_unpaired = cossim[B:B*2]
      vis_unpaired = cossim[B*2:]
      vis_rank_loss = self.vis_rank_weight * torch.clamp(self.margin + vis_unpaired - paired, min=0)
      text_rank_loss = self.text_rank_weight * torch.clamp(self.margin + text_unpaired - paired, min=0)
    
    loss = (vis_rank_loss + text_rank_loss).sum() / B
    
    return loss


class VisualGrounder(nn.Module):
  def __init__(self, 
               vis_size: int=2048,
               text_size: int=96,
               jemb_size: int= 2048,
               jemb_dropout: float = 0.,
               margin: float = 0.5,
               with_batch_norm: bool = False,
               load: Maybe[str] = None,
               vis_rank_weight = 1.,
               text_rank_weight = 1.
  ):
    super().__init__()
    self.crit = MaxMarginLoss(margin, vis_rank_weight, text_rank_weight)
    self.vis_size = vis_size
    self.text_size = text_size
    self.jemb_size = jemb_size
    self.batch_norm = nn.Identity() if not with_batch_norm else nn.BatchNorm1d(jemb_size)
    self.vis_mlp = nn.Sequential(nn.Linear(vis_size, jemb_size),
                                 self.batch_norm,
                                 nn.GELU(),
                                 nn.Dropout(p=jemb_dropout),
                                 nn.Linear(jemb_size, jemb_size),
                                 self.batch_norm
    )
    self.text_mlp = nn.Sequential(nn.Linear(text_size, jemb_size),
                                 self.batch_norm,
                                 nn.GELU(),
                                 nn.Dropout(p=jemb_dropout),
                                 nn.Linear(jemb_size, jemb_size),
                                 self.batch_norm
    )

    if load is not None:
      print(f'Loading visual grounder weights from {load}...')
      self.load_state_dict(torch.load(load))

  def forward(self, vis_input: Tensor, text_input: Tensor) -> Tensor:
    # vis: B x Dv, text: B x De, :batch_size->paired, batch_size: -> unpaired
    vis_emb = self.vis_mlp(vis_input)
    text_emb = self.text_mlp(text_input)
    vis_emb_norm = F.normalize(vis_emb, p=2, dim=1)
    text_emb_norm = F.normalize(text_emb, p=2, dim=1)
    cossim = torch.sum(vis_emb_norm * text_emb_norm, 1)
    return cossim.view(-1, 1) # B x 1

  @torch.no_grad()
  def predict_filter(self, vis_objects: Tensor, q: Tensor, unique: bool = False) -> Tensor:
    # vis_objects: N x Dv, q: De
    num_objects = vis_objects.shape[0]
    q_tile = q.unsqueeze(0).repeat(num_objects, 1)
    cossim = self.forward(vis_objects, q_tile)
    diff = (cossim - cossim.T - self.crit.margin) > 0
    if unique:
      # unique object -> maximum row-wise score
      return diff.sum(1).argmax(0).item() 
    else:
      # many objects -> pos row-wise sscore && non-zero col-wise score
      # rows = torch.argwhere(diff.sum(1)>0).squeeze().tolist()
      pos = torch.where(cossim.squeeze() > 0.25)[0].tolist()
      # pos = [pos] if isinstance(pos, int) else pos
      cols = torch.where(diff.sum(0)==0)[0].tolist()
      # cols = [cols] if isinstance(cols, int) else cols
      pos = list(set(pos).intersection(set(cols)))
      return pos
      #return cols
      # rows = [rows] if isinstance(rows, int) else rows
      # cols = [cols] if isinstance(cols, int) else cols
      # return list(set(rows).intersection(set(cols)))
      # return torch.argwhere(cossim.squeeze() > 0.15).squeeze().tolist()

  @torch.no_grad()
  def predict_query(self, vis_obj: Tensor, qs: Tensor) -> Tensor:
    # vis_obj: Dv, qs: M x De
    num_queries = qs.shape[0]
    vis_obj_tile = vis_obj.unsqueeze(0).repeat(num_queries, 1)
    cossim = self.forward(vis_obj_tile, qs)
    return cossim.squeeze().argmax().item()

  def compute_batch_metrics(self, cossim: Tensor) -> Dict[str, float]:
    batch_size = cossim.shape[0]
    vis_rank_correct, text_rank_correct = 0, 0
    B = 0 

    if self.crit.vis_rank and not self.crit.text_rank:
      print('vis_rank')
      B = batch_size // 2
      paired = cossim[:B]
      unpaired = cossim[B:]
      vis_rank_correct = (paired > unpaired).sum().item()
      text_rank_correct = 0

    elif not self.crit.vis_rank and self.crit.text_rank:
      print('txt_rank')
      B = batch_size // 2
      paired = cossim[:B]
      unpaired = cossim[B:]
      text_rank_correct = (paired > unpaired).sum().item()
      vis_rank_correct = 0

    elif self.crit.vis_rank and self.crit.text_rank:
      B = batch_size // 3
      paired = cossim[:B]
      text_unpaired = cossim[B:B*2]
      vis_unpaired = cossim[B*2:]
      vis_rank_correct = (paired > vis_unpaired).sum().item()
      text_rank_correct = (paired > text_unpaired).sum().item()
    
    return vis_rank_correct, text_rank_correct, B


  def train_step(self, dl: DataLoader, opt: Optimizer, scheduler: CosineAnnealingLR):
    self.train()
    epoch_loss, f_correct, q_correct, total = 0, 0, 0, 0
    for x, q in dl:
      cossim = self.forward(x, q) # B x 1
      opt.zero_grad()
      loss = self.crit(cossim)
      loss.backward()
      opt.step()
      scheduler.step()
      epoch_loss += loss.item()
      metrics = self.compute_batch_metrics(cossim)
      f_correct += metrics[0]
      q_correct += metrics[1]
      total += metrics[2]
    epoch_loss /= len(dl.dataset)
    metrics = {}
    if self.crit.vis_rank:
      metrics['filter'] = f_correct / total
    if self.crit.text_rank:
      metrics['query'] = q_correct / total
    return {'loss': epoch_loss, **metrics}

  @torch.no_grad()
  def eval_step(self, dl: DataLoader):
    self.eval()
    epoch_loss, f_correct, q_correct, total = 0, 0, 0, 0
    for x, q in dl:
      with torch.no_grad():
        cossim = self.forward(x, q) # B x 1 
        loss = self.crit(cossim)
        epoch_loss += loss.item()
        metrics = self.compute_batch_metrics(cossim)
      f_correct += metrics[0]
      q_correct += metrics[1]
      total += metrics[2]
    epoch_loss /= len(dl.dataset)
    metrics = {}
    if self.crit.vis_rank:
      metrics['filter'] = f_correct / total
    if self.crit.text_rank:
      metrics['query'] = q_correct / total
    return {'loss': epoch_loss, **metrics}
  
  def make_collate_fn(self, device: str) -> Map[List[Dict[str, Any]], Tuple[Tensor, ...]]:
      def _collate(batch: List[Dict[str, Any]]) -> Tuple[Tensor, ...]:
        vfeats = [item['vis_emb'] for item in batch] # [ N x Dv ] (B)
        vs_pos = torch.cat(vfeats, dim=0).to(device) # B*N x Dv
        qs_pos = torch.cat([item['q_emb_pos'] for item in batch], dim=0).to(device) # B*N x De
        qs_neg = torch.cat([item['q_emb_neg'] for item in batch], dim=0).to(device)
        vs_neg = torch.cat([vf[item['perm'],:] for vf, item in zip(vfeats, batch)],
          dim=0).to(device)

        if self.crit.vis_rank and not self.crit.text_rank:
          # final batch size is 2*B*N
          vs_batch = torch.cat([vs_pos, vs_neg], dim=0)
          qs_batch = torch.cat([qs_pos, qs_pos], dim=0)
        
        elif not self.crit.vis_rank and self.crit.text_rank:
          # final batch size is 2*B*N
          vs_batch = torch.cat([vs_pos, vs_pos], dim=0)
          qs_batch = torch.cat([qs_pos, qs_neg], dim=0)
        
        elif self.crit.vis_rank and self.crit.text_rank:
          # final batch size is 3*B*N
          vs_batch = torch.cat([vs_pos, vs_pos, vs_neg], dim=0)
          qs_batch = torch.cat([qs_pos, qs_neg, qs_pos], dim=0)
        
        return vs_batch, qs_batch
      
      return _collate