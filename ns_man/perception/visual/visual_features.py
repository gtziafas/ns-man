from ns_man.structs import *
from ns_man.utils.image_proc import crop_contour, crop_boxes_fixed

import torch
import torch.nn as nn 
import json
import cv2
import numpy as np


class VisualEmbedder(nn.Module):
    def __init__(self, 
                 feature_extractor: nn.Module,
                 preprocess: Map[List[array], List[array]],
                 device: str,
                 freeze: bool,
                 path: Maybe[str] = None
                 ):
        super().__init__()
        self.preprocess = preprocess
        self.device = device 
        self.feature_extractor = feature_extractor
        if path is not None:
            self.load_weights(path)
        if freeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor = self.feature_extractor.eval()

    def forward(self, x: Union[List[array], Tensor]) -> Tensor:
        if not isinstance(x, Tensor):
            x = self.tensorize(x)
        return self.feature_extractor.forward(x)

    def _tensorize(self, x: array) -> Tensor:
        x = torch.tensor(x/0xff, dtype=torch.float, device=self.device)
        return x.view(x.shape[-1], *x.shape[0:2])

    def tensorize(self, xs: List[array]) -> List[Tensor]:
        xs = xs if isinstance(xs, list) else [xs]
        xs = self.preprocess(xs)
        return list(map(self._tensorize, xs))

    @torch.no_grad()
    def features(self, xs: List[array]) -> Tensor:
        xs = torch.stack(self.tensorize(xs), dim=0)
        feats = self.feature_extractor(xs).flatten(1)
        return feats

    def load_weights(self, path: str):
        try:
            self.feature_extractor.load_state_dict(torch.load(path))
        except: 
            model = custom_classifier()
            model.load_state_dict(torch.load(path))
            self.feature_extractor = model.features.eval().to(self.device)


class CNNFeatures(nn.Module):
  def __init__(self, 
               num_blocks: int, 
               dropout_rates: List[float], 
               conv_kernels: List[int], 
               pool_kernels: List[int],
               input_channels: int = 3, 
               activation_fn: nn.Module = nn.LeakyReLU
               ):
      super().__init__()
      assert num_blocks == len(conv_kernels) == len(pool_kernels) == len(dropout_rates)
      self.activation_fn = activation_fn
      self.blocks = nn.Sequential(self.block(input_channels, 16, conv_kernels[0], pool_kernels[0], dropout_rates[0]),
                                  *[self.block(2**(3+i), 2**(4+i), conv_kernels[i], pool_kernels[i], dropout_rates[i])
                                  for i in range(1, num_blocks)])

  def block(self, 
            in_channels: int, 
            out_channels: int, 
            conv_kernel: int, 
            pool_kernel: int, 
            dropout: float = 0., 
            conv_stride: int = 1
           ):
      return nn.Sequential(
                           nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, stride=conv_stride),
                           self.activation_fn(),
                           nn.MaxPool2d(kernel_size=pool_kernel),
                           nn.Dropout(p=dropout)
                        ) 

  def forward(self, x: Tensor) -> Tensor:
      return self.blocks(x)


class CNNClassifier(nn.Module):
    def __init__(self, feature_extractor: CNNFeatures, head: nn.Module):
        super().__init__()
        self.features = feature_extractor
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        # x: B x 3 x H x W
        x = self.features(x).flatten(1) # B x D
        return self.head(x)


def custom_classifier(num_classes: int, params: Dict[str, Any], num_features: int) -> nn.Module:
    return CNNClassifier(
                            CNNFeatures(**params), 
                            nn.Linear(num_features, num_classes)
                        )


def custom_features(params) -> nn.Module:
    return CNNFeatures(**params)


def resnet18_classifier(num_classes: int, pretrained: bool, dropout: float = 0.5, load: Maybe[str] = None) -> nn.Module:
    from torchvision.models import resnet18
    model = resnet18(pretrained=pretrained)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
    if load is not None:
        model.load_state_dict(torch.load(load))
    return model


def resnet18_features(pretrained: bool, load: Maybe[str] = None) -> nn.Module:
    from torchvision.models import resnet18
    model = resnet18(pretrained=pretrained)
    model.fc = nn.Identity()
    if load is not None:
        model.load_state_dict(torch.load(load))
    return model

def resnet50_classifier(num_classes: int, pretrained: bool, dropout: float = 0.5, load: Maybe[str] = None) -> nn.Module:
    from torchvision.models import resnet50
    model = resnet50(pretrained=pretrained)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
    if load is not None:
        model.load_state_dict(torch.load(load))
    return model


def resnet50_features(pretrained: bool, load: Maybe[str] = None) -> nn.Module:
    from torchvision.models import resnet50
    model = resnet50(pretrained=pretrained)
    model.fc = nn.Identity()
    if load is not None:
        model.load_state_dict(torch.load(load))
    return model


def mobilenetv2_classifier(num_classes: int, pretrained: bool, dropout: float = 0.5, load: Maybe[str] = None) -> nn.Module:
    from torchvision.models import mobilenet_v2
    model = resnet18(pretrained=pretrained)
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))
    if load is not None:
        model.load_state_dict(torch.load(load))
    return model


def mobilenetv2_features(pretrained: bool, load: Maybe[str] = None) -> nn.Module:
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(pretrained=pretrained)
    model.classifier = nn.Identity()
    if load is not None:
        model.load_state_dict(torch.load(load))
    return model


def _tensorize(x: array) -> Tensor:
    x = torch.tensor(x/0xff, dtype=torch.float)
    return x.view(x.shape[-1], *x.shape[0:2])


def tensorize(xs: List[array]) -> Tensor:
    return torch.stack(list(map(_tensorize, xs)), dim=0)


# assuming images already pre-processed
def collate(device: str, labelset: Set[str]) -> Map[List[Tuple[array, str]], Tuple[Tensor, Tensor]]:
    idces = {v: k for k, v in enumerate(labelset)}
    def _collate(batch: List[Tuple[array, str]]) -> Tuple[Tensor, Tensor]:
        xs, ys = zip(*batch)
        xs = tensorize(xs).to(device)
        ys = torch.stack([torch.tensor(idces[y], dtype=longt, device=device) for y in ys], dim=0)
        return xs, ys
    return _collate


def make_visual_embedder(cfg: Union[str, Dict[str, Any]], device: str):
    cfg = json.load(open(cfg)) if isinstance(cfg, str) else cfg
    preprocess = crop_boxes_fixed(cfg['image_size'])
    version = cfg['model_version']
    if version == 'custom':
        print('Loading custom CNN model..')
        feature_extractor = custom_features(cfg['custom_cnn_params'])
    elif version == 'resnet18':
        print(f"Loading {'' if cfg['pretrained'] else 'non-'}pretrained {version} model..")
        feature_extractor = resnet18_features(cfg['pretrained'],
                                            cfg['load_checkpoint'])
    elif version == 'resnet50':
        print(f"Loading {'' if cfg['pretrained'] else 'non-'}pretrained {version} model..")
        feature_extractor = resnet50_features(cfg['pretrained'],
                                            cfg['load_checkpoint']                                           )
    elif version == 'mobilenetv2':
        print(f"Loading {'' if cfg['pretrained'] else 'non-'}pretrained {version} model..")
        feature_extractor = mobilenetv2_features(cfg['pretrained'],
                                                 cfg['load_checkpoint']
                                                )
    else:
        raise ValueError(f'uknown model version configuration {version}')

    path = cfg['load_checkpoint']
    return VisualEmbedder(feature_extractor, preprocess, device, path)