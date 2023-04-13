from ns_man.structs import *
from ns_man.programs.executor import HOTSExecutor
from ns_man.programs.tokenizer import postprocess_programs
from ns_man.language.seq2seq import make_seq2seq_net
from ns_man.language.parser import LanguageParser
from ns_man.language.word_embedding import make_word_embedder
from ns_man.grounders.spatial import RelationGrounder, HRelationGrounder
from ns_man.grounders.visual import VisualGrounder
from ns_man.grounders.modules import GroundersInterface
from ns_man.utils.scene_graph import *

import json
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm 
from collections import Counter
from einops import rearrange


SYNONYMS_PATH = "./data/synonyms/SynHOTS.json"
METADATA_PATH = "./data/metadata/SynHOTS.json"
SCENES_TRAIN_PATH = "./data/scenes/SynHOTS_scene_graphs_train.json"
SCENES_VAL_PATH = "./data/scenes/SynHOTS_scene_graphs_val.json"
SCENES_VEC_TRAIN_PATH = "./checkpoints/Perception/SynHOTS_scene_graphs_vec_train.p"
SCENES_VEC_VAL_PATH = "./checkpoints/Perception/SynHOTS_scene_graphs_vec_val.p"
QUESTIONS_TRAIN_PATH = "./data/questions/SynHOTS-VQA_proc_train.json"
QUESTIONS_VAL_PATH = "./data/questions/SynHOTS-VQA_proc_val.json"
QUESTIONS_TESTGEN_PATH = "./data/questions/SynHOTS-VQA_proc_TestGen_test.json"
VOCABS_PATH = "./checkpoints/Seq2seq/vocabs.json"
VOCABS_END2END_PATH = "./checkpoints/Seq2seq/vocabs_end2end.json"
VOCABS_TESTGEN_PATH = "./checkpoints/TestGen/vocabs.json"
VOCABS_END2END_TESTGEN_PATH = "./checkpoints/TestGen/vocabs_end2end.json"
VFEATS_RN50_CHP = "./checkpoints/Perception/SynHOTS_RN50_visual_features.p"
H, W = 480, 640


def get_question_type(program):
  terminal = program[-1]["type"]
  if terminal == 'count':
    return 'count'
  elif terminal == 'exist':
    return 'exist'
  elif terminal in ['equal_integer', 'greater_than', 'less_than']:
    return 'compare number'
  elif terminal in ['equal_material', 'equal_category', 'equal_color']:
    return 'compare attribute'
  elif terminal.startswith('query'):
    return 'query'


def unit_test_groundtruth_executor_nsvqa():
  executor = HOTSExecutor(METADATA_PATH,
              SYNONYMS_PATH,
              SCENES_TRAIN_PATH,
              SCENES_VAL_PATH,
              with_synonyms=False
  )
  ds_train = json.load(open(QUESTIONS_TRAIN_PATH))
  ds_val = json.load(open(QUESTIONS_VAL_PATH))
  for split,ds in [('train', ds_train), ('val', ds_val)]:
    for i, sample in enumerate(ds):
      p = postprocess_programs([sample['groundtruth_program']], reverse=False, tokenize=True)[0]
      a = sample['answer']
      assert isinstance(a, str)
      iid = sample['image_index']
      pred = executor.run_dataset(p, iid, split)
      assert pred == a
  print('Passed.')


def unit_test_seq2seq_executor_nsvqa():
  executor = HOTSExecutor(METADATA_PATH,
              SYNONYMS_PATH,
              SCENES_TRAIN_PATH,
              SCENES_VAL_PATH,
              with_synonyms=False
  )
  ds_val = json.load(open(QUESTIONS_VAL_PATH))
  WE = make_word_embedder()
  model = make_seq2seq_net("./config/seq2seq_rnn_end2end_cfg.json", load=True).eval().cuda()
  vocabs = json.load(open(VOCABS_END2END_PATH))
  vocabs['id2prog'] = {v:k for k,v in vocabs['prog2id'].items()}
  correct, total = {'overall': 0}, {'overall': 0}
  for sample in tqdm(ds_val):
    #p = postprocess_programs([sample['groundtruth_program']], reverse=False, tokenize=True)[0]
    q = WE(sample['tagger_input'])
    q_type = get_question_type(sample['groundtruth_program'])
    dummy = WE(" ".join(["<PAD>" ] * 30))
    txt_input = pad_sequence([dummy, q], batch_first=True, padding_value=model.pad_id)
    p = model.sample_output(txt_input.cuda())[1:].cpu().tolist() # remove dummy
    untils = [pt.index(model.end_id) if model.end_id in pt else len(p) for pt in p]
    program_tokens = [pt[:until][::-1] for pt, until in zip(p, untils)] # re-reverse
    program_tokens = [[vocabs['id2prog'][x] for x in pt] for pt in program_tokens]
    a = sample['answer']
    p = program_tokens[0]
    assert isinstance(a, str)
    iid = sample['image_index']
    pred = executor.run_dataset(p, iid, "val")
    total[q_type] = 1 if q_type not in total.keys() else total[q_type] + 1
    total['overall'] += 1
    if pred == a:
      correct[q_type] = 1 if q_type not in correct.keys() else correct[q_type] + 1
      correct['overall'] += 1
  results = {k: round(100 * v1/v2, 2) for (k,v1), (_,v2) in zip(correct.items(), total.items())}
  print(results)


def test_gen_seq2seq_executor_nsvqa():
  executor = HOTSExecutor(METADATA_PATH,
              SYNONYMS_PATH,
              SCENES_TRAIN_PATH,
              SCENES_VAL_PATH,
              with_synonyms=False
  )
  ds_val = json.load(open(QUESTIONS_TESTGEN_PATH))
  WE = make_word_embedder()
  model = make_seq2seq_net("./config/TestGen/seq2seq_rnn_end2end_cfg.json", load=True).eval().cuda()
  vocabs = json.load(open(VOCABS_END2END_TESTGEN_PATH))
  vocabs['id2prog'] = {v:k for k,v in vocabs['prog2id'].items()}
  correct, total = {'overall': 0}, {'overall': 0}
  for sample in tqdm(ds_val):
    #p = postprocess_programs([sample['groundtruth_program']], reverse=False, tokenize=True)[0]
    q = WE(sample['tagger_input'])
    #q_type = get_question_type(sample['groundtruth_program'])
    q_types = list(set(sample['gen_type']))
    dummy = WE(" ".join(["<PAD>" ] * 30))
    txt_input = pad_sequence([dummy, q], batch_first=True, padding_value=model.pad_id)
    p = model.sample_output(txt_input.cuda())[1:].cpu().tolist() # remove dummy
    untils = [pt.index(model.end_id) if model.end_id in pt else len(p) for pt in p]
    program_tokens = [pt[:until][::-1] for pt, until in zip(p, untils)] # re-reverse
    program_tokens = [[vocabs['id2prog'][x] for x in pt] for pt in program_tokens]
    a = sample['answer']
    p = program_tokens[0]
    assert isinstance(a, str)
    iid = sample['image_index']
    pred = executor.run_dataset(p, iid, "val")
    for q_type in q_types:
      total[q_type] = 1 if q_type not in total.keys() else total[q_type] + 1
    total['overall'] += 1
    if pred == a:
      for q_type in q_types:
        correct[q_type] = 1 if q_type not in correct.keys() else correct[q_type] + 1
      correct['overall'] += 1
  results = {k: round(100 * v1/v2, 2) for (k,v1), (_,v2) in zip(correct.items(), total.items())}
  print(results)

def unit_test_groundtruth_executor_nsman():
  executor = HOTSExecutor(METADATA_PATH,
              SYNONYMS_PATH,
              SCENES_TRAIN_PATH,
              SCENES_VAL_PATH,
              with_synonyms=True
  )
  ds_train = json.load(open(QUESTIONS_TRAIN_PATH))
  ds_val = json.load(open(QUESTIONS_VAL_PATH))
  wrong = {'train': [], 'val': []}
  for split,ds in [('train', ds_train), ('val', ds_val)]:
    for i, sample in enumerate(ds):
      p = []
      for j,node in enumerate(sample["seq2seq_output"]):
        tmp = [node['value_inputs'][0]] if node['value_inputs'] else []
        if node["value_inputs"]:
          x = node["value_inputs"][0]
          assert x in sample["concept_mapping"].keys(),  (x, sample["tagger_input"],  sample["tagger_output"], sample["seq2seq_input"], sample["concept_mapping"])
          y = sample["concept_mapping"][x]
          tmp = [y]
        p.append({'type':node['type'], 'value_inputs':tmp})
      pp = postprocess_programs([p], reverse=False, tokenize=True)[0]
      a = sample['answer']
      assert isinstance(a, str)
      iid = sample['image_index']
      pred = executor.run_dataset(pp, iid, split)
      #assert pred == a, (i, pred, a, sample["seq2seq_input_single"], sample["tagger_input"], pp, [n['type'] for n in sample['groundtruth_program']])
      if pred != a:
        wrong[split].append(i)
  print(wrong)
  #print('Passed.')


def unit_test_seq2seq_executor_nsman(groundtruth_tags = False):
  executor = HOTSExecutor(METADATA_PATH,
              SYNONYMS_PATH,
              SCENES_TRAIN_PATH,
              SCENES_VAL_PATH,
              with_synonyms=True
  )
  ds_val = json.load(open(QUESTIONS_VAL_PATH))
  parser = LanguageParser({'use_bert_tagger':False, 
    'device': 'cuda', 
    'vocabs_path': VOCABS_PATH, 
    'bert_checkpoint': './checkpoints/Tag/bert/pytorch_model.bin',
    'tagger_rnn_cfg': "./config/tagger_rnn_cfg.json"}
  )
  correct, total = {'overall': 0}, {'overall': 0}; crashed=0
  for sample in tqdm(ds_val):
    q_type = get_question_type(sample['groundtruth_program'])
    if groundtruth_tags:
      # tags -> programs
      tagged_input, tagged_replaced, concept_map = sample['seq2seq_input_single'], sample['seq2seq_input'], sample['concept_mapping']
      p = parser._generate_program([tagged_input.split()], [tagged_replaced.split()], [concept_map])[0]
    else:
      # language -> tags -> programs
      q = sample['tagger_input']
      try:
        p = parser.parse(q)
      except:
        # print(q, parser.tag(q))
        # break
        crashed +=1
        continue  
    a = sample['answer']
    assert isinstance(a, str)
    iid = sample['image_index']
    pred = executor.run_dataset(p, iid, "val")
    total[q_type] = 1 if q_type not in total.keys() else total[q_type] + 1
    total['overall'] += 1
    if pred == a:
      correct[q_type] = 1 if q_type not in correct.keys() else correct[q_type] + 1
      correct['overall'] += 1
  results = {k: round(100 * v1/v2, 2) for (k,v1), (_,v2) in zip(correct.items(), total.items())}
  print(results)
  print(crashed, crashed/len(ds_val))


def test_gen_seq2seq_executor_nsman(groundtruth_tags = False):
  executor = HOTSExecutor(METADATA_PATH,
              SYNONYMS_PATH,
              SCENES_TRAIN_PATH,
              SCENES_VAL_PATH,
              with_synonyms=True
  )
  ds_val = json.load(open(QUESTIONS_TESTGEN_PATH))
  parser = LanguageParser({'use_bert_tagger':False, 
    'device': 'cuda', 
    'vocabs_path': VOCABS_TESTGEN_PATH, 
    'bert_checkpoint': './checkpoints/Tag/bert/pytorch_model.bin',
    'tagger_rnn_cfg': "./config/TestGen/tagger_rnn_cfg.json"}
  )
  correct, total = {'overall': 0}, {'overall': 0}; crashed=0
  for sample in tqdm(ds_val):
    q_types = sample['gen_type']
    if groundtruth_tags:
      # tags -> programs
      tagged_input, tagged_replaced, concept_map = sample['seq2seq_input_single'], sample['seq2seq_input'], sample['concept_mapping']
      p = parser._generate_program([tagged_input.split()], [tagged_replaced.split()], [concept_map])[0]
    else:
      # language -> tags -> programs
      q = sample['tagger_input']
      try:
        p = parser.parse(q)
      except:
        # print(q, parser.tag(q))
        # break
        crashed +=1
        continue  
    a = sample['answer']
    assert isinstance(a, str)
    iid = sample['image_index']
    pred = executor.run_dataset(p, iid, "val")
    for q_type in q_types:
      total[q_type] = 1 if q_type not in total.keys() else total[q_type] + 1
    total['overall'] += 1
    if pred == a:
      for q_type in q_types:
        correct[q_type] = 1 if q_type not in correct.keys() else correct[q_type] + 1
      correct['overall'] += 1
  results = {k: round(100 * v1/v2, 2) for (k,v1), (_,v2) in zip(correct.items(), total.items())}
  print(results)
  print(crashed, crashed/len(ds_val))


def unit_test_relation_grounder_ft():
  # load GloVe embedding
  WE = make_word_embedder('glove_sm')

  #synonyms = json.load(open(synonyms_path))
  scenes_val = json.load(open(SCENES_VAL_PATH))['scenes']
  
  model = RelationGrounder(load='checkpoints/Grounders/spatial/relation_grounder_ft_h96j9_balanced_90k_weights.p').cuda().eval()
  
  synonyms = REL_SYNONYMS
  all_rel_words= sum(REL_SYNONYMS.values(), [])
  all_rel_embs = {q: WE(q).mean(0) for q in all_rel_words}

  pair_correct, scene_correct, pair_total, scene_total = 0, 0, 0, 0
  rel_correct = {k.lower(): 0 for k in RELATIONS}
  rel_total = {k.lower(): 0 for k in RELATIONS}
  for scene_graph in scenes_val:
    num_objects = len(scene_graph['objects'])
    x_rel = torch.from_numpy(vectorize_edges(scene_graph, skip_hrel=True)).cuda() # N x N x R
    
    _all_correct = True
    for rel_idx, rel in enumerate(RELATIONS):
      rel = rel.lower()
      choices = synonyms[rel]
      query = random.choice(choices)  
      query_emb = all_rel_embs[query].cuda() # De
      out = model.predict(x_rel, query_emb) # N x N
      labels = x_rel[:, :, rel_idx].bool().cpu().numpy() # N x N

      pair_correct += (out == labels).sum().item()
      pair_total += num_objects**2

      if np.allclose(out, labels):
        rel_correct[rel] += 1
      else:
        _all_correct = False
      rel_total[rel] += 1

    if _all_correct:
      scene_correct += 1
    scene_total += 1

  pair_accu = 100 * pair_correct / pair_total
  rel_accu = {k: round(100 * v1/ v2, 2) for (k,v1),(_,v2) in zip(rel_correct.items(), rel_total.items())}
  scene_accu = 100 * scene_correct / scene_total
  print(pair_correct, pair_total, rel_correct, rel_total, scene_correct, scene_total)
  print(f"Pair accuracy = {pair_accu:.2f}, Relation accuracy = {rel_accu}, Scene accuracy={scene_accu:.2f}")



def unit_test_relation_grounder():
  scenes_val = json.load(open(SCENES_VAL_PATH))['scenes']
  # synonyms = json.load(open(SYNONYMS_PATH))
  # synonyms = {k: v for k,v in synonyms.items() if k.upper() in RELATIONS}
  synonyms = REL_SYNONYMS
  dividend = torch.as_tensor([W, H, W, H, W, H, W, H], dtype=torch.float)
  dividend.required_grad = False
  WE = make_word_embedder()
  model = RelationGrounder(with_text_proj=True, load='checkpoints/Grounders/spatial/relation_grounder_ft_j9_sigma_balanced_90k_weights.p').cuda().eval()
  pair_correct, scene_correct, pair_total, scene_total = 0, 0, 0, 0
  rel_correct = {k.lower(): 0 for k in RELATIONS}
  rel_total = {k.lower(): 0 for k in RELATIONS}
  for scene_graph in tqdm(scenes_val):
    num_objects = len(scene_graph['objects'])
    pair_dists = torch.from_numpy(compute_pairwise_dist(scene_graph['objects'])).float()
    # N x 5 spatial features
    spt_feats_2d_c = [torch.as_tensor(o['RGB_center'], dtype=torch.float) / dividend[:2] for o in scene_graph['objects']]
    spt_feats_2d_r = [torch.as_tensor(o['RGB_rectangle'], dtype=torch.float) / dividend for o in scene_graph['objects']]
    #spt_feats_3d = [torch.as_tensor(normalize_coords(o['position_3d'][:3]), dtype=torch.float) for o in scene_graph['objects']]
    spt_feats_size = [torch.as_tensor([o['size'] / SIZE_NORM]) for o in scene_graph['objects']]
    #spt = torch.stack([torch.cat([x1,x2,x3,x4], dim=0) for x1,x2,x3,x4 in zip(spt_feats_2d_c, spt_feats_2d_r, spt_feats_3d, spt_feats_size)]).cuda()
    spt = torch.stack([torch.cat([x1,x2,x3], dim=0) for x1,x2,x3 in zip(spt_feats_2d_c, spt_feats_2d_r, spt_feats_size)]).cuda()
    # N x N x 10 spatial-pair features
    ys = vectorize_edges(scene_graph, skip_hrel=True)
    queries = []
    for rel in rel_correct.keys():
      # sample query relation synonym
      choices = synonyms[rel]
      queries.append(random.choice(choices))
    query_emb = torch.stack([WE(q).mean(0) for q in queries]).cuda() # R x De
    spt_tile = spt.unsqueeze(0).repeat(query_emb.shape[0], 1, 1) # R x N x Dx
    pair_dists_tile = pair_dists.unsqueeze(0).repeat(query_emb.shape[0], 1, 1).cuda()
    pred = model.predict(spt_tile, pair_dists_tile, query_emb).astype(bool) # R x N x N
    _all_correct = True
    assert ys.shape[-1] == pred.shape[0]
    for rel_idx, p in enumerate(list(pred)):
      rel = RELATIONS[rel_idx].lower()
      truth = ys[:,:,rel_idx].astype(bool)
      np.fill_diagonal(p, False)
      rel_total[rel] += 1
      pair_correct += (p == truth).sum().item()
      pair_total += num_objects**2
      if (p == truth).sum().item() == num_objects**2:
        rel_correct[rel] += 1
      else:
        _all_correct = False
    if _all_correct:
      scene_correct += 1
    scene_total += 1
  pair_accu = 100 * pair_correct / pair_total
  rel_accu = {k: round(100 * v1/ v2, 2) for (k,v1),(_,v2) in zip(rel_correct.items(), rel_total.items())}
  scene_accu = 100 * scene_correct / scene_total
  print(pair_correct, pair_total, rel_correct, rel_total, scene_correct, scene_total)
  print(f"Pair accuracy = {pair_accu:.2f}, Relation accuracy = {rel_accu}, Scene accuracy={scene_accu:.2f}")


def unit_test_visual_grounder_query():
  ds_val = torch.load(VFEATS_RN50_CHP)['val']
  metadata = json.load(open(METADATA_PATH))
  # synonyms = json.load(open(SYNONYMS_PATH))
  # synonyms = {k: v for k,v in synonyms.items() if k.upper() in RELATIONS}
  model = VisualGrounder(load='checkpoints/Grounders/visual/visual_grounder_3_weights.p').cuda().eval()
  
  WE = make_word_embedder('glove_sm')

  concept_embeddings, concept_classes ={}, {}
  for concept in ['Color', 'Material', 'Category']:
      concept_classes[concept] = metadata['types'][concept]
      concept_embeddings[concept] = torch.stack([WE(x).mean(0) for x in concept_classes[concept]]).cuda()

  obj_correct, obj_total = 0, 0
  concept_correct = {k: 0 for k in ['Color', 'Material', 'Category']}
  concept_total = {k: 0 for k in ['Color', 'Material', 'Category']}
  scene_correct, scene_total = 0, 0

  for _,sample in tqdm(ds_val.items()):
    
    obj_vfeats = sample['visual_features'].cuda()
    obj_names = [x[0] for x in sample['attribute_labels']]
    attr_labels = [{k: x for k, x in zip(['Color', 'Material', 'Category'], xs[1:])} for xs in sample['attribute_labels']]

    _all_correct = True

    for i in range(len(obj_names)):
      
      v = obj_vfeats[i]
      name = obj_names[i]
      labels = attr_labels[i]

      pred_v = {}
      for concept in concept_classes.keys():
        pred_id = model.predict_query(v, concept_embeddings[concept])
        pred = concept_classes[concept][pred_id]
        pred_v[concept] = pred

      if pred_v == labels:
        obj_correct += 1
      else:
        _all_correct = False
      obj_total += 1

      for (concept, p), (_, l) in zip(pred_v.items(), labels.items()):
        if p == l:
          concept_correct[concept] += 1
        concept_total[concept] += 1

    if _all_correct:
      scene_correct += 1
    scene_total += 1

  obj_accu = 100 * obj_correct / obj_total
  scene_accu = 100 * scene_correct / scene_total
  conc_accu = {k: v1/v2 for (k,v1),(_,v2) in zip(concept_correct.items(), concept_total.items())}
  print(f"object accu={obj_accu:.2f}%, concepts = {conc_accu}, scene accu={scene_accu:.2f}%")



def unit_test_visual_grounder_filter():
  ds_val = torch.load(VFEATS_RN50_CHP)['val']
  metadata = json.load(open(METADATA_PATH))
  synonyms = json.load(open(SYNONYMS_PATH))
  # synonyms = {k: v for k,v in synonyms.items() if k.upper() in RELATIONS}
  model = VisualGrounder(load='checkpoints/Grounders/visual/visual_grounder_3_weights.p').cuda().eval()
  
  WE = make_word_embedder('glove_sm')

  unique_correct, many_correct, many_total = 0, 0, 0

  for _,sample in tqdm(ds_val.items()):
    
    obj_vfeats = sample['visual_features'].cuda()
    obj_names = [x[0] for x in sample['attribute_labels']]
    attr_labels = [x[1:] for x in sample['attribute_labels']]

    all_labels_cnt = dict(Counter(sum(attr_labels, [])))
    
    q_unique = random.choice([k for k,v in all_labels_cnt.items() if v == 1])
    truth_unique = [i for i,ls in enumerate(attr_labels) if q_unique in ls]
    assert len(truth_unique) == 1
    truth_unique = truth_unique[0]
    q_unique_syn = random.choice(synonyms[q_unique])
    q_unique_emb = WE(q_unique_syn).mean(0).cuda()
    
    out_unique = model.predict_filter(obj_vfeats, q_unique_emb, unique=True)
    
    if truth_unique == out_unique:
      unique_correct += 1

    choose_from = [k for k,v in all_labels_cnt.items() if v > 1]
    if not choose_from:
      continue
    q_many = random.choice(choose_from)
    truth_many = [i for i, ls in enumerate(attr_labels) if q_many in ls]
    q_many_syn = random.choice(synonyms[q_many])
    q_many_emb = WE(q_many_syn).mean(0).cuda()
    
    out_many = model.predict_filter(obj_vfeats, q_many_emb)

    if truth_many == out_many:
      many_correct += 1
    many_total += 1

  unique_accu = 100 * unique_correct / len(ds_val)
  many_accu = 100 * many_correct / many_total
  print(f"filter unique accu={unique_accu:.2f}%, filter many accu={many_accu:.2f}%")


def unit_test_hrel_grounder():
  scenes_val = json.load(open(SCENES_VAL_PATH))['scenes']
  synonyms = HREL_SYNONYMS
  dividend = torch.as_tensor([W, H, W, H, W, H, W, H], dtype=torch.float)
  WE = make_word_embedder()
  model = HRelationGrounder(load='checkpoints/Grounders/spatial/hrel_grounder_200k_2_weights.p').cuda().eval()
  pair_correct, scene_correct, pair_total, scene_total = 0, 0, 0, 0
  rel_correct = {k.lower(): 0 for k in HRELATIONS}
  rel_total = {k.lower(): 0 for k in HRELATIONS} 
  for scene_graph in tqdm(scenes_val):
    num_objects = len(scene_graph['objects'])
    centers = torch.stack([torch.as_tensor(normalize_coords(o['position_3d']), 
      dtype=torch.float) for o in scene_graph['objects']]).cuda()

    pair_dists = torch.from_numpy(compute_pairwise_dist(scene_graph['objects'])).float()
    pair_dist_tile = pair_dists.unsqueeze(2).repeat(1, 1, num_objects)
    delta_x = pair_dist_tile - pair_dist_tile.transpose(1, 2) # N x N x N
    y = {}
    y['closer'] = (delta_x < 0).numpy()
    y['further'] = (delta_x > 0).numpy()

    _all_correct = True
    for k in HRELATIONS:
        k = k.lower()
        q = random.choice(synonyms[k])
        q_emb = WE(q).mean(0).cuda()
        rel_total[k] += 1

        out = model.predict_scene(centers.unsqueeze(0), q_emb.unsqueeze(0)) # N x N x N
        if np.allclose(out, y[k]):
          rel_correct[k] += 1
        else:
          _all_correct = False

        pair_total += num_objects**3
        pair_correct += (out == y[k]).sum().item()
    
    if _all_correct:
      scene_correct += 1
    scene_total += 1

  pair_accu = 100 * pair_correct / pair_total
  rel_accu = {k: round(100 * v1/ v2, 2) for (k,v1),(_,v2) in zip(rel_correct.items(), rel_total.items())}
  scene_accu = 100 * scene_correct / scene_total
  print(pair_correct, pair_total, rel_correct, rel_total, scene_correct, scene_total)
  print(f"Pair accuracy = {pair_accu:.2f}, Relation accuracy = {rel_accu}, Scene accuracy={scene_accu:.2f}")


def unit_test_concept_grounder():
  GI = GroundersInterface("config/concept_grounding_interface.json")
  synonyms = json.load(open(SYNONYMS_PATH))
  scenes_val = torch.load(SCENES_VEC_VAL_PATH)
  wrong_r = []
  wrong_fu = 0 ; total_fu = 0
  wrong_f = 0 ; total_f = 0
  i=0
  wrong_s = 0
  all_hrel_words = sum(HREL_SYNONYMS.values(), [])
  all_rel_words = sum(REL_SYNONYMS.values(), [])
  all_hrel_embs = {q: GI.WE(q).mean(0) for q in all_hrel_words}
  all_rel_embs = {q: GI.WE(q).mean(0) for q in all_rel_words}
  all_name_embs = {q: GI.WE(q).mean(0) for q in sum(synonyms.values(), [])}
  GI.with_glove = False
  for g in tqdm(scenes_val):
      GI.set(g)
      _all = True

      obj_names = [x[0] for x in g['attribute_labels']]
      attr_labels = [x[1:] for x in g['attribute_labels']]

      for j, name in enumerate(obj_names):
        if name not in synonyms.keys():
          continue
        q = random.choice(synonyms[name])
        qemb = all_name_embs[q]

        pred = GI.filter_unique(list(range(g['n_objects'])), qemb)
        assert isinstance(pred, int)
        
        if j != pred:
          wrong_fu += 1
        total_fu += 1


      for j, attrs in enumerate(attr_labels):
        q = random.choice(attrs)
        qsyn = random.choice(synonyms[q]) if q in synonyms.keys() else q
        qemb = all_name_embs[qsyn]

        truth = [idx for idx in range(len(attr_labels)) if q in attr_labels[idx]]

        preds = GI.filter(list(range(g['n_objects'])), qemb)
        assert isinstance(preds, list)

        if preds != truth:
          wrong_f += 1
        total_f += 1

      for k, v in g['scene_graph_symbols']['relationships'].items():
          
          if k in ['closer','further']:
              q = random.choice(HREL_SYNONYMS[k])
              qemb = all_hrel_embs[q]
              pred = [[GI.hyper_relate(n, m, qemb) for m in range(g['n_objects'])] for n in range(g['n_objects'])]
              #continue
          
          else:
              q = random.choice(REL_SYNONYMS[k])
              qemb = all_rel_embs[q]
              pred = [GI.relate(n, qemb) for n in range(g['n_objects'])]

          if v != pred:
              wrong_r.append([i,k])
              _all = False
      if not _all:
          wrong_s += 1
      i += 1
  vis_accu = 0.5 * ((total_f - wrong_f) / total_f + (total_fu - wrong_fu) / total_fu)
  scene_accu = (len(scenes_val) - wrong_s) / len(scenes_val)
  rel_accu = (len(scenes_val)*11 - len(wrong_r)) / (len(scenes_val) * 11)
  print(f'Rel+HRel accuracy={100*rel_accu:.2f}%, visual accuracy={100*vis_accu:.2f}%, scene accuracy={100*scene_accu:.2f}%')
