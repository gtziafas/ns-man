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

Metrics = Dict[str, Any]



class MSELossIgnore(nn.Module):
    def __init__(self, ignore_index: int, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.core = nn.MSELoss(**kwargs)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        predictions = predictions[targets.ne(self.ignore_index)]
        targets = targets[targets.ne(self.ignore_index)]
        return self.core(predictions, targets)



class BCEWithLogitsIgnore(nn.Module):
    def __init__(self, ignore_index: int, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.core = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        predictions = predictions[targets.ne(self.ignore_index)]
        targets = targets[targets.ne(self.ignore_index)]
        return self.core(predictions, targets)


class HRelationGrounder(nn.Module):

  def __init__(self, 
               spt_size: int = 1,
               text_size: int = 96, 
               hidden_size: int = 96,
               jemb_size: int = 2,
               jemb_dropout: float = 0.,
               with_text_proj: bool = True,
               load: Maybe[str] = None
               ):
        super().__init__()
        self.jemb_size = jemb_size
        self.spt_size = spt_size
        self.text_size = text_size
        self.spt_proj_1 = nn.Linear(spt_size, hidden_size).apply(self.init_weights)
        # self.spt_proj_1 = nn.Linear(1 + 2 * spt_size, hidden_size).apply(self.init_weights)
        self.spt_proj_2 = nn.Linear(hidden_size, jemb_size).apply(self.init_weights)
        self.text_proj_1 = nn.Linear(text_size, hidden_size).apply(self.init_weights) if with_text_proj else nn.Identity()
        self.text_proj_2 = nn.Linear(hidden_size, jemb_size).apply(self.init_weights) if with_text_proj else nn.Identity()
        self.dropout = nn.Dropout(p=jemb_dropout)
        self.match = nn.Linear(jemb_size, 1)
        self.with_text_proj = with_text_proj
        self.ignore_idx = -100

        if load is not None:
          print(f'Loading relation grounder weights from {load}...')
          self.load_state_dict(torch.load(load))

  def forward_step(self, delta_x: Tensor, q: Tensor) -> Tensor:
    # delta_x: B x H, query: B x De
    x_proj = F.gelu(self.spt_proj_1(delta_x))
    x_proj = self.spt_proj_2(x_proj)
    q_proj = F.gelu(self.text_proj_1(q))
    q_proj = self.text_proj_2(q_proj)
    jembs = F.normalize(x_proj * q_proj, p=2, dim=-1)
    out = self.match(self.dropout(jembs))
    return out

  @staticmethod
  def pairwise_dist(centers_3d: Tensor) -> Tensor:
    return torch.cdist(centers_3d, centers_3d, p=2)

  def init_weights(self, m: nn.Module):
      if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
          torch.nn.init.kaiming_uniform_(m.weight)
          m.bias.data.fill_(0.01)

  @torch.no_grad()
  def predict(self, delta_x: Tensor, q: Tensor) -> array:
    # delta_x: N x N x N x 1, q: De
    num_objects = delta_x.shape[0]
    delta_x_flat = rearrange(delta_x, "n m k h -> (n m k) h")
    q_tile = q.unsqueeze(0).repeat(num_objects**3, 1)
    out = self.forward_step(delta_x_flat, q_tile)
    return out.view(num_objects, num_objects, num_objects).sigmoid().ge(0.5).cpu().numpy()

  def forward_scene(self, pos_embs: Tensor, query: Tensor) -> Tensor:
    # pos_embs: B x N x 3, query: B x De
    batch_size, num_objects = pos_embs.shape[0:2]

    dst = self.pairwise_dist(pos_embs) # B x N x N
    dst_tile = dst.unsqueeze(3).repeat(1, 1, 1, num_objects) # B x N x N x N
    delta_x = (dst_tile - dst_tile.transpose(2,3)).unsqueeze(4) # B x N x N x N x 1
    
    q_tile = query.unsqueeze(1).repeat(1, num_objects**3, 1) # B x N x N x N x De
    q_tile = q_tile.view(batch_size, num_objects, num_objects, num_objects, -1)

    out = self.forward_step(delta_x, q_tile) # B x N x N x N x1
    return out

  @torch.no_grad()
  def predict_scene(self, pos_embs: Tensor, q_emb: Tensor) -> array:
    # pair_dists = self.pairwise_dist(spt_inputs)
    return self.forward_scene(pos_embs, q_emb).squeeze().sigmoid().ge(0.5).bool().cpu().numpy()

  def init_weights(self, m: nn.Module):
      if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
          torch.nn.init.kaiming_uniform_(m.weight)
          m.bias.data.fill_(0.01)

  def make_collate_fn(self, device: str) -> Map[List[Dict[str, Any]], Tuple[Tensor, ...]]:
      def _collate(batch: List[Dict[str, Any]]) -> Tuple[Tensor, ...]:
        xs = torch.stack([item['x'] for item in batch]).unsqueeze(1).float().to(device)
        qs = torch.stack([item['q_emb'] for item in batch]).float().to(device)
        ys = torch.stack([item['label'] for item in batch]).float().to(device)
        return xs, qs, ys
      return _collate

  @torch.no_grad()
  def compute_metrics(self, predictions: Tensor, truth: Tensor, ignore_idx: int = -1) -> Dict[str, float]:
    accuracy = accuracy_score(truth, predictions)
    precision = precision_score(truth, predictions)
    recall = recall_score(truth, predictions)
    f1 = f1_score(truth, predictions)
    return {'acc': accuracy, 'p': precision, 'r': recall, 'f1': f1}

  def train_step(self, dl: DataLoader, crit: nn.Module, opt: Optimizer, scheduler: CosineAnnealingLR) -> Metrics:
    self.train()
    epoch_loss, all_preds, all_truth = 0., [], []
    for batch_idx, (xs, qs, ys) in enumerate(dl):
      pred = self.forward_step(xs, qs) # [B, 1]
      opt.zero_grad()
      loss = crit(pred.sigmoid().squeeze(-1), ys.float())
      loss.backward()
      opt.step()
      scheduler.step()
      epoch_loss += loss.item()
      #correct += (pred.flatten().sigmoid().ge(0.5).long() == ys.flatten()).sum().item()
      all_preds.extend(pred.flatten().sigmoid().ge(0.5).float().tolist())
      all_truth.extend(ys.flatten().tolist())
    metrics = self.compute_metrics(all_preds, all_truth)
    epoch_loss /= len(dl.dataset)
    return {**{'loss': epoch_loss}, **metrics}

  @torch.no_grad()
  def eval_step(self, dl: DataLoader, crit: nn.Module) -> Tuple[float, ...]:
    self.eval()
    epoch_loss, all_preds, all_truth = 0., [], []
    for batch_idx, (xs, qs, ys) in enumerate(dl):
      with torch.no_grad():
        pred = self.forward_step(xs, qs) # [B, 1]
        loss = crit(pred.sigmoid().squeeze(-1), ys.float())
        epoch_loss += loss.item()
        #correct += (pred.flatten().sigmoid().ge(0.5).long() == ys.flatten()).sum().item()
        all_preds.extend(pred.flatten().sigmoid().ge(0.5).float().tolist())
        all_truth.extend(ys.flatten().tolist())
    metrics = self.compute_metrics(all_preds, all_truth)
    epoch_loss /= len(dl.dataset)
    return {**{'loss': epoch_loss}, **metrics}

    
class RelationGrounder(nn.Module):

  def __init__(self, 
               spt_size: int = 9,
               text_size: int = 96, 
               hidden_size: int = 96,
               jemb_size: int = 9,
               jemb_dropout: float = 0.,
               with_text_proj: bool = True,
               load: Maybe[str] = None
               ):
        super().__init__()
        self.jemb_size = jemb_size
        self.spt_size = spt_size
        self.text_size = text_size
        self.spt_proj_1 = nn.Linear(spt_size, hidden_size).apply(self.init_weights)
        # self.spt_proj_1 = nn.Linear(1 + 2 * spt_size, hidden_size).apply(self.init_weights)
        self.spt_proj_2 = nn.Linear(hidden_size, jemb_size).apply(self.init_weights)
        self.text_proj_1 = nn.Linear(text_size, hidden_size).apply(self.init_weights) if with_text_proj else nn.Identity()
        self.text_proj_2 = nn.Linear(hidden_size, jemb_size).apply(self.init_weights) if with_text_proj else nn.Identity()
        self.dropout = nn.Dropout(p=jemb_dropout)
        self.match = nn.Linear(jemb_size, 1)
        self.with_text_proj = with_text_proj
        self.ignore_idx = -100

        if load is not None:
          print(f'Loading relation grounder weights from {load}...')
          self.load_state_dict(torch.load(load))

  def forward_step(self, x_pair: Tensor, q: Tensor) -> Tensor:
    # x_pair: B x 2*Dx, query: B x De
    x_proj = self.spt_proj_2(F.gelu(self.spt_proj_1(x_pair)))
    # x_proj = F.gelu(self.spt_proj_1(x_pair))
    # x_proj = x_pair
    q_proj = F.gelu(self.text_proj_1(q))
    # q_proj = torch.sigmoid(self.text_proj_1(q))
    q_proj = self.text_proj_2(q_proj)
    # x_proj = self.spt_proj(x_pair).tanh()
    # q_proj = self.text_proj(q).tanh()
    # x_proj = self.spt_proj(x_pair)
    # q_proj = self.text_proj(q)
    jembs = F.normalize(x_proj * q_proj, p=2, dim=-1)
    out = self.match(self.dropout(jembs))
    # out = (x_proj * q_proj).sum(1, keepdim=True)
    return out

  def forward_scene(self, spt_embs: Tensor, pair_dists: Tensor, query: Tensor) -> Tensor:
    # spt_embs: B x N x Dx, query: B x De
    batch_size, num_objects = spt_embs.shape[0:2]
    
    # N x N pair-wise position embeddings, flattened
    x_tile = spt_embs.unsqueeze(2).repeat(1, 1, num_objects, 1) # B x N x N x Dx
    x_pair = torch.cat([x_tile, x_tile.transpose(1,2), pair_dists.unsqueeze(3)], dim=3)  
    #x_pair = torch.cat([x_tile - x_tile.transpose(1,2), pair_dists.unsqueeze(3)], dim=3)  
    #x_pair = x_pair.view(batch_size * num_objects**2, -1) # B * N^2 x 2*Dx+1
    #x_pair = rearrange(x_pair, "b n m d -> (b n m) d")
    
    q_tile = query.unsqueeze(1).repeat(1, num_objects**2, 1)
    q_tile = q_tile.view(batch_size, num_objects, num_objects, -1)

    out = self.forward_step(x_pair, q_tile) # B x N x N x 1
    #out = out.squeeze(-1).contiguous().view(batch_size, num_objects, num_objects)
    #out = rearrange(out.squeeze(-1), '(b n m) -> b n m')

    return out.squeeze(-1)

  @torch.no_grad()
  def predict(self, x_pair: Tensor, q_emb: Tensor) -> array:
    # x_pair: N x N x R, # q_emb: De
    num_objects = x_pair.shape[0]
    q_emb_tile = q_emb.unsqueeze(0).repeat(num_objects**2, 1)
    x_pair_flat = rearrange(x_pair, "n m r -> (n m) r")
    out = self.forward_step(x_pair_flat, q_emb_tile) # N x N x 1
    out = out.view(num_objects, num_objects)
    return out.sigmoid().ge(0.5).cpu().numpy() 

  @torch.no_grad()
  def predict_scene(self, spt_inputs: Tensor, pair_dists: Tensor, q_emb: Tensor) -> array:
    # pair_dists = self.pairwise_dist(spt_inputs)
    return self.forward_scene(spt_inputs, pair_dists, q_emb).squeeze().sigmoid().ge(0.5).bool().cpu().numpy()

  def init_weights(self, m: nn.Module):
      if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
          torch.nn.init.kaiming_uniform_(m.weight)
          m.bias.data.fill_(0.01)

  def make_collate_fn(self, device: str) -> Map[List[Dict[str, Any]], Tuple[Tensor, ...]]:
      def _collate(batch: List[Dict[str, Any]]) -> Tuple[Tensor, ...]:
        xs = torch.stack([item['x_pair'] for item in batch]).float().to(device)
        qs = torch.stack([item['q_emb'] for item in batch]).float().to(device)
        ys = torch.stack([item['label'] for item in batch]).float().to(device)
        return xs, qs, ys
      return _collate

  def make_collate_fn_seq(self, device: str) -> Map[List[Dict[str, Any]], Tuple[Tensor, ...]]:
      def _collate(batch: List[Dict[str, Any]]) -> Tuple[Tensor, ...]:
        # xs: B x N x Dx, xs_d: B x N x N, qs: B x Dq, ys: B x N x N
        xs = pad_sequence([item['x_emb'] for item in batch], batch_first=True, padding_value=0)
        maxlen = max([item['x_dist'].shape[0] for item in batch])
        xs_d = torch.stack([F.pad(item['x_dist'], pad=(0, maxlen - item['x_dist'].shape[0], 0, maxlen - item['x_dist'].shape[1]), 
          mode="constant", value=0) for item in batch]) # B x N x N
        qs = torch.stack([item['q_emb'] for item in batch],)
        ys = pad_sequence([item['label'].flatten() for item in batch], 
          batch_first=True, padding_value=self.ignore_idx) # B x N^2
        return xs.to(device), xs_d.to(device), qs.to(device), ys.to(device)
      return _collate

  @torch.no_grad()
  def compute_metrics(self, predictions: Tensor, truth: Tensor, ignore_idx: int = -1) -> Dict[str, float]:
    accuracy = accuracy_score(truth, predictions)
    precision = precision_score(truth, predictions)
    recall = recall_score(truth, predictions)
    f1 = f1_score(truth, predictions)
    return {'acc': accuracy, 'p': precision, 'r': recall, 'f1': f1}

  @torch.no_grad()
  def compute_metrics_padded(self, predictions: Tensor, truth: Tensor) -> Dict[str, float]:
    predictions = predictions[truth != self.ignore_idx]
    truth = truth[truth != self.ignore_idx]
    accuracy = accuracy_score(truth, predictions)
    precision = precision_score(truth, predictions)
    recall = recall_score(truth, predictions)
    f1 = f1_score(truth, predictions)
    return {'acc': accuracy, 'p': precision, 'r': recall, 'f1': f1}

  @torch.no_grad()
  def get_confusion_matrix(self, predictions: Tensor, truth: Tensor) -> Tensor:
    total_pos = truth[truth==1].shape[0]
    total_neg = truth[truth==0].shape[0]
    true_pos = predictions[truth==1].sum().item()
    true_neg = (~predictions[truth==0].bool()).sum().item()
    false_pos = torch.where(torch.bitwise_and(predictions==1, truth==0), 1, 0).sum().item()
    false_neg = torch.where(torch.bitwise_and(predictions==0, truth==1), 1, 0).sum().item()
    return torch.tensor([[true_pos, true_neg], [false_pos, false_neg], [total_pos, total_neg]])

  def get_metrics_from_matrix(self, matrix: Tensor) -> Metrics:
    TP, TN, FP, FN, P, N = matrix.flatten().tolist()
    return {"acc": round((TP+TN) / (P+N), 4),
            "tpr": round(TP / P, 4),
            "p": round(TP / (TP+FP+1e-9), 4),
            "r":   round(TP / (TP+FN+1e-9), 4),
            "f1": round(2*TP / (2*TP+FP+FN+1e-9), 4)
            }
  
  def train_step(self, dl: DataLoader, crit: nn.Module, opt: Optimizer, scheduler: CosineAnnealingLR) -> Metrics:
    self.train()
    epoch_loss, all_preds, all_truth = 0., [], []
    for batch_idx, (xs, qs, ys) in enumerate(dl):
      pred = self.forward_step(xs, qs) # [B, 1]
      opt.zero_grad()
      loss = crit(pred.sigmoid(), ys.float())
      loss.backward()
      opt.step()
      scheduler.step()
      epoch_loss += loss.item()
      #correct += (pred.flatten().sigmoid().ge(0.5).long() == ys.flatten()).sum().item()
      all_preds.extend(pred.flatten().sigmoid().ge(0.5).float().tolist())
      all_truth.extend(ys.flatten().tolist())
    metrics = self.compute_metrics(all_preds, all_truth)
    epoch_loss /= len(dl.dataset)
    return {**{'loss': epoch_loss}, **metrics}

  @torch.no_grad()
  def eval_step(self, dl: DataLoader, crit: nn.Module) -> Tuple[float, ...]:
    self.eval()
    epoch_loss, all_preds, all_truth = 0., [], []
    for batch_idx, (xs, qs, ys) in enumerate(dl):
      with torch.no_grad():
        pred = self.forward_step(xs, qs) # [B, 1]
        loss = crit(pred.sigmoid(), ys.float())
        epoch_loss += loss.item()
        #correct += (pred.flatten().sigmoid().ge(0.5).long() == ys.flatten()).sum().item()
        all_preds.extend(pred.flatten().sigmoid().ge(0.5).float().tolist())
        all_truth.extend(ys.flatten().tolist())
    metrics = self.compute_metrics(all_preds, all_truth)
    epoch_loss /= len(dl.dataset)
    return {**{'loss': epoch_loss}, **metrics}

  def train_step_seq(self, dl: DataLoader, crit: nn.Module, opt: Optimizer, scheduler: CosineAnnealingLR) -> Metrics:
    self.train()
    epoch_loss, all_preds, all_truth = 0., [], []
    confusion_matrix = torch.zeros(3, 2)
    confusion_matrix.requires_grad = False
    for batch_idx, (xs, xsd, qs, ys) in enumerate(dl):
      pred = self.forward(xs, xsd, qs) # [B, N, N]
      opt.zero_grad()
      loss = crit(pred.flatten(1), ys)
      loss.backward()
      opt.step()
      scheduler.step()
      epoch_loss += loss.item()
      #all_preds.extend(pred.flatten().sigmoid().ge(0.5).float().tolist())
      #all_truth.extend(ys.flatten().tolist())
      confusion_matrix += self.get_confusion_matrix(pred.flatten(1).ge(0.5).bool(), ys.bool())
    metrics = self.get_metrics_from_matrix(confusion_matrix)
    #metrics = self.compute_metrics_padded(np.array(all_preds), np.array(all_truth))
    epoch_loss /= len(dl.dataset)
    return {**{'loss': epoch_loss}, **metrics}

  @torch.no_grad()
  def eval_step_seq(self, dl: DataLoader, crit: nn.Module) -> Tuple[float, ...]:
    self.eval()
    epoch_loss, all_preds, all_truth = 0., [], []
    confusion_matrix = torch.zeros(3, 2)
    for batch_idx, (xs, xsd, qs, ys) in enumerate(dl):
      with torch.no_grad():
        pred = self.forward(xs, xsd, qs) # [B, N, N]
        loss = crit(pred.flatten(1), ys)
        epoch_loss += loss.item()
        #all_preds.extend(pred.flatten().sigmoid().ge(0.5).float().tolist())
        #all_truth.extend(ys.flatten().tolist())
        confusion_matrix += self.get_confusion_matrix(pred.flatten(1).ge(0.5).bool(), ys.bool())
    metrics = self.get_metrics_from_matrix(confusion_matrix)
    #metrics = self.compute_metrics_padded(np.array(all_preds), np.array(all_truth))
    epoch_loss /= len(dl.dataset)
    return {**{'loss': epoch_loss}, **metrics}