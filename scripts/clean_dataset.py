import os
import json


def cleanings(ds):
    skip=[]
    for i, sample in enumerate(ds):
        if '<Y> <Y>' in sample['seq2seq_input_single']:
            skip.append(i)


    skip=[]
    for i, sample in enumerate(ds):
        Map = {'filter_category' : '<Y>', 'filter_color': '<C>', 'filter_material': '<M>',
            'locate': '<L>', 'relate': '<R>', 'hyper_relate': '<H>'}
        Map_inv = {v:k for k,v in Map.items()}
        tmp = {'filter_category' : 0, 'filter_color': 0, 'filter_material': 0,
            'locate': 0, 'relate': 0, 'hyper_relate': 0}
        tmp1 = tmp.copy()
        cm = sample['concept_mapping']
        for k, v in cm.items():
            key = k if len(k)  ==3 else k[:-2] + '>'
            #print(key)
            tmp[Map_inv[key]] += 1
        for node in sample['groundtruth_program']:
            fn = node['type']
            fn = 'filter_category' if fn =='ground' else fn
            if fn in Map.keys():
                tmp1[fn] += 1
        if tmp != tmp1:
            skip.append(i)


def make_test_gen(train_path, val_path, vocabulary):
  # vocabulary contains {'category': [unseen cat words], 'color': [unseen color words], etc.}
  ds_train = json.load(open(train_path))
  ds_val = json.load(open(val_path))
  all_unseen = sum(list(vocabulary.values()), [])
  
  ds_train_keep = []
  ds_test = []
  for sample in ds_train:
    if any(w in sample['tagger_input'] for w in all_unseen):
      continue
    ds_train_keep.append(sample)

  appended = set()
  for i, sample in enumerate(ds_val):
    for k, v in vocabulary.items():
      for w in v:
        if w in sample['tagger_input']:
          if i not in appended:
            appended.add(i)
            ds_test.append(sample)
          if "gen_type" not in ds_test[i].keys():
            ds_test[i]["gen_type"] = [k]
          else:
            ds_test[i]["gen_type"].append(k)

  return ds_train_keep, ds_test

