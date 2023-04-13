from ns_man.structs import *
from ns_man.perception.visual.visual_features import make_visual_embedder

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


class MLP(nn.Module):
  def __init__(self, 
               hidden_sizes: List[int],
               dropout_rates: List[float],
               num_classes: int,
               input_dim: int,
               activation_fn: nn.Module = nn.GELU
        ):
      super().__init__()
      assert (len(hidden_sizes) + 1) == len(dropout_rates)
      self.activation_fn = activation_fn
      self.num_hidden_layers = len(hidden_sizes)
      self.input_dropout = nn.Dropout(dropout_rates[0])
      self.layers = nn.ModuleList([self.layer(input_dim, hidden_sizes[0], dropout_rates[1])])
      self.layers.extend([self.layer(hidden_sizes[i], hidden_sizes[i+1], dropout_rates[i+2]) 
          for i in range(self.num_hidden_layers-1)])
      self.layers = nn.Sequential(*[*self.layers, nn.Linear(hidden_sizes[-1], num_classes)])

  def layer(self, inp_dim: int, out_dim: int, dropout: float) -> nn.Module:
      return nn.Sequential(
                            nn.Linear(inp_dim, out_dim),
                            self.activation_fn(),
                            nn.Dropout(dropout)
                        )

  def forward(self, x: Tensor) -> Tensor:
      x = self.input_dropout(x)
      return self.layers(x)


class BaseMLPHead(nn.Module):
    def __init__(self, attribute: str, config: Dict[str, Any]):
        super().__init__()
        if attribute not in ['color', 'material', 'category']:
            raise ValueError(f'Unknown attribute {attribute}')
        
        self.config = config
        self.embedding_size = config['embedding_size']
        
        if config['load_features_checkpoint'] is None:
            self.feature_extractor = config['feature_extractor']
            freeze = config['freeze_feature_extractor']
            assert freeze is not None, f'Set {freeze} to True/False if using feature extractor backbone'
            if freeze:
                for param in self.feature_extractor:
                    param.required_grad = False
                self.feature_extractor = feature_extractor.eval()
        else:
            self.feature_extractor = nn.Identity()

        self.head = MLP(**config[f'{attribute}_head_params'])
        
    def forward(self, x: Tensor) -> Tensor:
        # assert we are properly loading feature checkpoint
        assert (len(x.shape) == 2 and self.config['load_features_checkpoint'] and x.shape[-1] == self.embedding_size) or \
               (len(x.shape) == 4 and self.config['load_features_checkpoint'] == False)

        x = self.feature_extractor(x)
        x = self.head(x)
        return x


class AttributeHeads(nn.Module):
    def __init__(self, config: Union[str, Dict[str, Any]]):
        super().__init__()
        self.cfg = json.load(open(config)) if isinstance(config, str) else config
        self.maps ={'color': 0, 'material': 1, 'category': 2}
        self.all_heads = nn.ModuleList([BaseMLPHead('color', self.cfg),
                                     BaseMLPHead('material', self.cfg),
                                     BaseMLPHead('category', self.cfg)])

        if self.cfg['load_checkpoint'] is not None:
            print(f'Loading visual modules from {self.cfg["load_checkpoint"]}')
            self.load_state_dict(torch.load(self.cfg['load_checkpoint']))

    @property
    def color_head(self):
        return self.all_heads[self.maps['color']]
    
    @property
    def material_head(self):
        return self.all_heads[self.maps['material']]
    
    @property
    def category_head(self):
        return self.all_heads[self.maps['category']]

    def forward(self, x: Tensor) -> Tensors:
        return [head(x) for head in self.all_heads]

    def make_collate_fn(self, device: str) -> Map[List[Tensors], Tensors]:
        def _collate_fn(batch: List[Tensors]) -> Tensors:
            batch = zip(*batch)
            return [torch.stack(data).to(device) for data in batch]
        return _collate_fn

    Metrics = Dict[str, Any]

    def train_step(self, dl: DataLoader, optim: Optimizer, crit: nn.Module) -> Metrics:
        self.train()
        
        epoch_loss = 0.
        attr_losses = {k: 0. for k in self.maps.keys()}
        attr_correct= {k: 0 for k in self.maps.keys()}

        for vfeats, cols, mats, cats in dl:
            out = self.forward(vfeats)

            col_loss = crit(out[self.maps['color']], cols)
            mat_loss = crit(out[self.maps['material']], mats)
            cat_loss = crit(out[self.maps['category']], cats)

            # backprop
            optim.zero_grad()
            losses = [self.cfg['loss_weights']['color'] * col_loss,
                     self.cfg['loss_weights']['material'] * mat_loss,
                     self.cfg['loss_weights']['category'] * cat_loss
                     ]
            sum(losses).backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
            optim.step()

            # update
            epoch_loss += sum(losses).item()
            attr_losses = {k: v + l.item() for (k, v), l in zip(attr_losses.items(), losses)}
            attr_correct = {k : v + (out[self.maps[k]].argmax(-1) == labels).sum().item() \
                for (k, v), labels in zip(attr_correct.items(), [cols, mats, cats])}
 
        epoch_loss /= len(dl)
        attr_losses = {k: v / len(dl) for k,v in attr_losses.items()}
        attr_correct = {k: v / len(dl.dataset) for k, v in attr_correct.items()}
        
        return {**{'total_loss': epoch_loss},
                **{k + '_loss': v for k, v in attr_losses.items()},
                **{k + '_accuracy': v for k, v in attr_correct.items()}
                }

    @torch.no_grad()
    def eval_step(self, dl: DataLoader, crit: nn.Module) -> Metrics:
        self.eval()
        
        epoch_loss = 0.
        attr_losses = {k: 0. for k in self.maps.keys()}
        attr_correct= {k: 0 for k in self.maps.keys()}

        for vfeats, cols, mats, cats in dl:
            out = self.forward(vfeats)
            
            col_loss = crit(out[self.maps['color']], cols)
            mat_loss = crit(out[self.maps['material']], mats)
            cat_loss = crit(out[self.maps['category']], cats)
            losses = [self.cfg['loss_weights']['color'] * col_loss,
                     self.cfg['loss_weights']['material'] * mat_loss,
                     self.cfg['loss_weights']['category'] * cat_loss
                     ]
            
            # update
            epoch_loss += sum(losses).item()
            attr_losses = {k: v + l.item() for (k, v), l in zip(attr_losses.items(), losses)}
            attr_correct = {k : v + (out[self.maps[k]].argmax(-1) == labels).sum().item() \
                for (k, v), labels in zip(attr_correct.items(), [cols, mats, cats])}
 
        epoch_loss /= len(dl)
        attr_losses = {k: v / len(dl) for k,v in attr_losses.items()}
        attr_correct = {k: v / len(dl.dataset) for k, v in attr_correct.items()}
        
        return {**{'total_loss': epoch_loss},
                **{k + '_loss': v for k, v in attr_losses.items()},
                **{k + '_accuracy': v for k, v in attr_correct.items()}
                }


class AttributeClassifier(nn.Module):
    
    def __init__(self, config: Union[str, Dict[str, Any]]
                ):
        super().__init__()
        self.cfg = json.load(open(config)) if isinstance(config, str) else config
        num_classes = self.cfg['num_classes']
        
        self.maps = {'color': 0, 'material': 1, 'category': 2}
        self.backbone = resnet18(pretrained=self.cfg['pretrained'])
        for p in self.backbone.fc.parameters():
            p.requires_grad = False

        self.color_head = nn.Sequential(nn.Linear(512, 64), nn.GELU(), nn.Dropout(0.1), 
            nn.Linear(64, num_classes['color'])).apply(self.init_weights)
        self.material_head = nn.Sequential(nn.Linear(512, 64), nn.GELU(), nn.Dropout(0.1), 
            nn.Linear(64, num_classes['material'])).apply(self.init_weights)
        self.category_head = nn.Sequential(nn.Linear(512, 64), nn.GELU(), nn.Dropout(0.1), 
            nn.Linear(64, num_classes['category'])).apply(self.init_weights)

        if self.cfg['load'] is not None:
            print('Loading attribute predictor from checkpoint..')
            self.load_state_dict(torch.load(self.cfg['load']))

    def init_weights(self, net: nn.Module):
        if isinstance(net, nn.Linear):
            nn.init.xavier_uniform_(net.weight)
            net.bias.data.fill_(0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        pooled = self.backbone.avgpool(x).flatten(1)
        cx = self.color_head(pooled)
        mx = self.material_head(pooled)
        yx = self.category_head(pooled)

        return cx, mx, yx

    def make_collate_fn(self, device: str) -> Map[List[Tensors], Tensors]:
        def _collate_fn(batch: List[Tensors]) -> Tensors:
            batch = zip(*batch)
            return [torch.stack(data).to(device) for data in batch]
        return _collate_fn

    Metrics = Dict[str, Any]

    def train_step(self, dl: DataLoader, optim: Optimizer, crit: nn.Module) -> Metrics:
        self.train()
        
        epoch_loss = 0.
        attr_losses = {k: 0. for k in self.maps.keys()}
        attr_correct= {k: 0 for k in self.maps.keys()}

        for vfeats, cols, mats, cats in dl:
            out = self.forward(vfeats)
            col_preds, mat_preds, cat_preds = out

            col_loss = crit(col_preds, cols)
            mat_loss = crit(mat_preds, mats)
            cat_loss = crit(cat_preds, cats)

            # backprop
            optim.zero_grad()
            losses = [self.cfg['loss_weights']['color'] * col_loss,
                     self.cfg['loss_weights']['material'] * mat_loss,
                     self.cfg['loss_weights']['category'] * cat_loss
                     ]
            sum(losses).backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
            optim.step()

            # update
            epoch_loss += sum(losses).item()
            attr_losses = {k: v + l.item() for (k, v), l in zip(attr_losses.items(), losses)}
            attr_correct = {k : v + (out[self.maps[k]].argmax(-1) == labels).sum().item() \
                for (k, v), labels in zip(attr_correct.items(), [cols, mats, cats])}
 
        epoch_loss /= len(dl)
        attr_losses = {k: v / len(dl) for k,v in attr_losses.items()}
        attr_correct = {k: v / len(dl.dataset) for k, v in attr_correct.items()}
        
        return {**{'total_loss': epoch_loss},
                **{k + '_loss': v for k, v in attr_losses.items()},
                **{k + '_accuracy': v for k, v in attr_correct.items()}
                }

    @torch.no_grad()
    def eval_step(self, dl: DataLoader, crit: nn.Module) -> Metrics:
        self.eval()
        
        epoch_loss = 0.
        attr_losses = {k: 0. for k in self.maps.keys()}
        attr_correct= {k: 0 for k in self.maps.keys()}

        for vfeats, cols, mats, cats in dl:
            with torch.no_grad():
                out = self.forward(vfeats)
                col_preds, mat_preds, cat_preds = out 
                
                col_loss = crit(col_preds, cols)
                mat_loss = crit(mat_preds, mats)
                cat_loss = crit(cat_preds, cats)

                losses = [self.cfg['loss_weights']['color'] * col_loss,
                         self.cfg['loss_weights']['material'] * mat_loss,
                         self.cfg['loss_weights']['category'] * cat_loss
                         ]
            
            # update
            epoch_loss += sum(losses).item()
            attr_losses = {k: v + l.item() for (k, v), l in zip(attr_losses.items(), losses)}
            attr_correct = {k : v + (out[self.maps[k]].argmax(-1) == labels).sum().item() \
                for (k, v), labels in zip(attr_correct.items(), [cols, mats, cats])}
 
        epoch_loss /= len(dl)
        attr_losses = {k: v / len(dl) for k,v in attr_losses.items()}
        attr_correct = {k: v / len(dl.dataset) for k, v in attr_correct.items()}
        
        return {**{'total_loss': epoch_loss},
                **{k + '_loss': v for k, v in attr_losses.items()},
                **{k + '_accuracy': v for k, v in attr_correct.items()}
                }
