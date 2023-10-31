from ns_man.structs import *
from ns_man.grounders.spatial import RelationGrounder, HRelationGrounder
from ns_man.grounders.visual import VisualGrounder
from ns_man.language.word_embedding import make_word_embedder

import json
import numpy as np
import torch

Obj = int 
ObjSet = Sequence[int]
 

class GroundersInterface:
  def __init__(self, cfg: Union[str, Dict[str, Any]]):
      self.cfg = json.load(open(cfg)) if isinstance(cfg, str) else cfg
      self.device = self.cfg['device']
      self.with_glove = self.cfg['with_glove']
      if self.with_glove:
        self.WE = make_word_embedder('glove_sm')
      
      self.relG = RelationGrounder(load=self.cfg['rel_grounder_weights_path']).eval().to(self.device)
      self.visG = VisualGrounder(load=self.cfg['visual_grounder_weights_path']).eval().to(self.device)
      self.hrelG = HRelationGrounder(load=self.cfg['hrel_grounder_weights_path']).eval().to(self.device) 

      # self.concept_embeddings, self.concept_classes = {}, {}
      #   for concept in ['Color', 'Material', 'Category']:
      #       self.concept_classes[concept] = self.metadata['types'][concept]
    
      self.concept_memory = torch.load(self.cfg['concept_memory_path'])
      self.concept_classes = self.concept_memory['concept_classes']
      self.concept_embeddings = self.concept_memory['concept_embeddings']

  def set(self, scene_graph: Dict[str, Any]):
      # scene: obj_feats (N x Dv), rel_feats (N x N x R), hrel_feats (N x N x N x H)
      self.obj_feats = scene_graph['obj_feats'].to(self.device)
      self.rel_feats = scene_graph['rel_feats'].to(self.device)
      self.hrel_feats = scene_graph['hrel_feats'].unsqueeze(3).to(self.device)
      self.n_objects = scene_graph['n_objects']
      self.attribute_labels = scene_graph['attribute_labels']

  def relate(self, m: Obj, q: Union[str, Tensor])  -> ObjSet:
      if self.with_glove:
        q = self.WE(q).mean(0)
      pair_scores = self.relG.predict(self.rel_feats, q.to(self.device)) # N x N
      out_ids = np.where(pair_scores[:, m])[0].tolist()
      return [out_ids] if isinstance(out_ids, int) else out_ids

  def locate(self, V: ObjSet, q: Union[str, Tensor]) -> Obj:
      if self.with_glove:
        q = self.WE(q).mean(0)
      input_object_ids = {k: v for k, v in enumerate(V)}
      pair_scores = self.relG.predict(self.rel_feats, q.to(self.device)) # N x N
      scores = pair_scores[V, :][:, V]
      return input_object_ids[scores.sum(1).argmax()]

  def hyper_relate(self, m1: Obj, m2: Obj, q: Union[str, Tensor]) -> ObjSet:
      if self.with_glove:
        q = self.WE(q).mean(0)
      pair_scores = self.hrelG.predict(self.hrel_feats, q.to(self.device)) # N x N x N
      pair_scores = pair_scores[:, m1, :].T[m2, :]
      out_ids = [i for i, x in enumerate(pair_scores) if x and i not in [m1, m2]]
      return out_ids
    
  def filter(self, V: ObjSet, q: Union[str, Tensor]) -> ObjSet:
      if self.with_glove:
        q = self.WE(q).mean(0)
      input_object_ids = {k: v for k, v in enumerate(V)}
      input_objects = self.obj_feats[V, :]
      assert input_objects.shape[0] == len(V)
      out_ids = self.visG.predict_filter(input_objects, q.to(self.device), unique=False)
      out_ids = [out_ids] if isinstance(out_ids, int) else out_ids
      return [input_object_ids[i] for i in out_ids]

  def filter_unique(self, V: ObjSet, q: Union[str, Tensor]) -> Obj:
      if self.with_glove:
        q = self.WE(q).mean(0)
      if len(V) == 0:
        # empty set, run for scene
        V = list(range(self.n_objects))
      input_object_ids = {k: v for k, v in enumerate(V)}
      input_objects = self.obj_feats[V, :]
      out_id = self.visG.predict_filter(input_objects, q.to(self.device), unique=True)
      return input_object_ids[out_id]

  def query_color(self, n: Obj,) -> str:
      qs = torch.stack([self.concept_embeddings[k] for k in self.concept_classes['Color']])
      obj_feats = self.obj_feats[n]
      value_id = self.visG.predict_query(obj_feats, qs.to(self.device))
      return self.concept_classes['Color'][value_id]

  def query_material(self, n: Obj,) -> str:
      qs = torch.stack([self.concept_embeddings[k] for k in self.concept_classes['Material']])
      obj_feats = self.obj_feats[n]
      value_id = self.visG.predict_query(obj_feats, qs.to(self.device))
      return self.concept_classes['Material'][value_id]

  def query_category(self, n: Obj,) -> str:
      qs = torch.stack([self.concept_embeddings[k] for k in self.concept_classes['Category']])
      obj_feats = self.obj_feats[n]
      value_id = self.visG.predict_query(obj_feats, qs.to(self.device))
      return self.concept_classes['Category'][value_id]


def make_concept_grounders(cfg: Maybe[Union[str, Dict[str, Any]]]=None):
    cfg = cfg or "./config/concept_grounding_interface.json"
    cfg = json.load(open(cfg)) if isinstance(cfg, str) else cfg
    return GroundersInterface(cfg)


# class VGInterface:
#   ''' A vision grounder - only interface - relations are handled from scene graphs '''

#   def __init__(self, cfg: Union[str, Dict[str, Any]]):
#       self.cfg = json.load(open(cfg)) if isinstance(cfg, str) else cfg
#       self.device = self.cfg['device']
#       self.with_glove = self.cfg['with_glove']
#       if self.with_glove:
#         self.WE = make_word_embedder('glove_sm')
      
#       self.visG = VisualGrounder(load=self.cfg['visual_grounder_weights_path']).eval().to(self.device)
      
#       self.concept_memory = torch.load(self.cfg['concept_memory_path'])
#       self.concept_classes = self.concept_memory['concept_classes']
#       self.concept_embeddings = self.concept_memory['concept_embeddings']

#   def set(self, scene_graph: Dict[str, Any]):
#       # scene: obj_feats (N x Dv), rel_feats (N x N x R), hrel_feats (N x N x N x H)
#       self.obj_feats = scene_graph['obj_feats'].to(self.device)
#       self.n_objects = scene_graph['n_objects']
#       self.attribute_labels = scene_graph['attribute_labels']
    
#   def filter(self, V: ObjSet, q: Union[str, Tensor]) -> ObjSet:
#       if self.with_glove:
#         q = self.WE(q).mean(0)
#       input_object_ids = {k: v for k, v in enumerate(V)}
#       input_objects = self.obj_feats[V, :]
#       out_ids = self.visG.predict_filter(input_objects, q.to(self.device), unique=False)
#       out_ids = [out_ids] if isinstance(out_ids, int) else out_ids
#       return [input_object_ids[i] for i in out_ids]

#   def filter_unique(self, V: ObjSet, q: Union[str, Tensor]) -> Obj:
#       if self.with_glove:
#         q = self.WE(q).mean(0)
#       if len(V) == 0:
#         # empty set, run for scene
#         V = list(range(self.n_objects))
#       input_object_ids = {k: v for k, v in enumerate(V)}
#       input_objects = self.obj_feats[V, :]
#       out_id = self.visG.predict_filter(input_objects, q.to(self.device), unique=True)
#       return input_object_ids[out_id]

#   def query_color(self, n: Obj,) -> str:
#       qs = torch.stack([self.concept_embeddings[k] for k in self.concept_classes['Color']])
#       obj_feats = self.obj_feats[n]
#       value_id = self.visG.predict_query(obj_feats, qs.to(self.device))
#       return self.concept_classes['Color'][value_id]

#   def query_material(self, n: Obj,) -> str:
#       qs = torch.stack([self.concept_embeddings[k] for k in self.concept_classes['Material']])
#       obj_feats = self.obj_feats[n]
#       value_id = self.visG.predict_query(obj_feats, qs.to(self.device))
#       return self.concept_classes['Material'][value_id]

#   def query_category(self, n: Obj,) -> str:
#       qs = torch.stack([self.concept_embeddings[k] for k in self.concept_classes['Category']])
#       obj_feats = self.obj_feats[n]
#       value_id = self.visG.predict_query(obj_feats, qs.to(self.device))
#       return self.concept_classes['Category'][value_id]