from ns_man.structs import *
from ns_man.utils.image_proc import *
from ns_man.utils.load_dataset import get_sim_rgbd_scenes
from ns_man.perception.visual_features import make_visual_embedder

import json
from tqdm import tqdm
import torch


def make_attribute_prediction_dataset(json_train_path: str,
                     json_val_path: str,
                    save: Maybe[str] = None,
                    device: str = 'cpu'
) -> Dict[str, Dict[[int, Any]]]:
  # loading scene graphs
  scenes_train = json.load(open(json_train_path))['scenes']
  scenes_val = json.load(open(json_val_path))['scenes']
  
  # loading image data
  ds_train, ds_val = get_sim_rgbd_scenes()

  result = {}
  for (split, ds, scene_graphs) in [('train', ds_train, scenes_train), ('val', ds_val, scenes_val)]:
      ds = []
      print(f'Doing {split}...')
      for scene_graph in tqdm(scene_graphs):
        scene = ds.get_from_id(scene_graph['image_index'])
        img = ds.get_image_from_id(scene_graph['image_index'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crops = [crop_contour(img, o.contour) for o in scene.objects]
        attr_labels = [[o['color'], o['material'], o['category']] for o in scene_graph['objects']]
        ds.extend([(img, *labels) for img,labels in zip(crops, attr_labels)])
        
      result[split] = ds
    if save is not None:
    	torch.save(result, save)
    return result


def extract_visual_features_from_dataset( 
                     json_train_path: str,
                     json_val_path: str,
                    config_path: str,
                    save: Maybe[str] = None,
                    device: str = 'cpu'
) -> Dict[str, Dict[[int, Any]]]:

  # loading scene graphs
  scenes_train = json.load(open(json_train_path))['scenes']
  scenes_val = json.load(open(json_val_path))['scenes']
  
  # loading image data
  ds_train, ds_val = get_sim_rgbd_scenes()

  # pretrained visual feature extractor network
  VE = make_visual_embedder(config_path, device)

  all_feats = {}
  for (split, ds, scene_graphs) in [('train', ds_train, scenes_train), ('val', ds_val, scenes_val)]:
      feats = {}
      print(f'Doing {split}...')
      for scene_graph in tqdm(scene_graphs):
        scene = ds.get_from_id(scene_graph['image_index'])
        img = ds.get_image_from_id(scene_graph['image_index'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crops = [crop_contour(img, o.contour) for o in scene.objects]
        feats[scene_graph['image_index']] = {'visual_features': VE.features(crops),
        	'scene_graph': scene_graph,
        	'attribute_labels': [[o['label'], o['color'], o['material'], o['category']] for o in scene_graph['objects']]
        }
      all_feats[split] = feats

  if save is not None:
      torch.save(all_feats, save)

  return all_feats


