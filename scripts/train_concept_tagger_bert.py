import torch
import torch.nn as nn 
import torch.nn.functional as F
from math import ceil
import numpy as np 
import json
import evaluate
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer,EarlyStoppingCallback
import random


TAGSET = [
    '0',
    'B-CAT',
    'B-COL',
    'B-HREL',
    'B-LOC',
    'B-MAT',
    'B-REL',
    'I-CAT',
    'I_COL',
    'I-HREL',
    'I-LOC',
    'I-MAT',
    'I-REL'
 ]


def main(n_data, batch_size, save):

	tagging_dataset_path_train = './data/questions/SynHOTS-VQA_proc_train.json'
	tagging_dataset_path_dev = './data/questions/SynHOTS-VQA_proc_val.json'

	# load datasets
	ds_train = json.load(open(tagging_dataset_path_train))
	ds_dev = json.load(open(tagging_dataset_path_dev))
	tagset = TAGSET
	tag2id = {v:k for k,v in enumerate(sorted(tagset))}
	id2tag = {k:v for k,v in enumerate(sorted(tagset))}

	tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

	# align sub-word tokens with tags
	def tokenize_and_align_labels(samples):
	    tokenized_inputs = tokenizer(samples['x'], truncation=True, is_split_into_words=True)

	    labels = []
	    for i, label in enumerate(samples['y']):
	        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
	        previous_word_idx = None
	        label_ids = []
	        for word_idx in word_ids:  # Set the special tokens to -100.
	            if word_idx is None:
	                label_ids.append(-100)
	            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
	                label_ids.append(label[word_idx])
	            else:
	                label_ids.append(-100)
	            previous_word_idx = word_idx
	        labels.append(label_ids)

	    tokenized_inputs["labels"] = labels
	    return tokenized_inputs


	ds_train = random.sample(ds_train, n_data) if n_data is not None else ds_train
	ds_train_tokens = tokenize_and_align_labels({'x': [x["tagger_input"].split() for x in ds_train], "y": [[tag2id[t] for t in x['tagger_output'].split()] for x in ds_train]})
	ds_dev_tokens = tokenize_and_align_labels({'x': [x["tagger_input"].split() for x in ds_dev], "y": [[tag2id[t] for t in x['tagger_output'].split()] for x in ds_dev]})
	ds_train_tokens = [{'input_ids':x, 'attention_mask':m, 'labels':y} for x,m,y in zip(ds_train_tokens["input_ids"], ds_train_tokens["attention_mask"], ds_train_tokens["labels"])]
	ds_dev_tokens = [{'input_ids':x, 'attention_mask':m, 'labels':y} for x,m,y in zip(ds_dev_tokens["input_ids"], ds_dev_tokens["attention_mask"], ds_dev_tokens["labels"])]

	data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

	# labels = [label_list[i] for i in example[f"ner_tags"]]
	seqeval = evaluate.load("seqeval")

	def compute_metrics(p):
	    predictions, labels = p
	    predictions = np.argmax(predictions, axis=2)

	    true_predictions = [
	        [tagset[p] for (p, l) in zip(prediction, label) if l != -100]
	        for prediction, label in zip(predictions, labels)
	    ]
	    true_labels = [
	        [tagset[l] for (p, l) in zip(prediction, label) if l != -100]
	        for prediction, label in zip(predictions, labels)
	    ]

	    results = seqeval.compute(predictions=true_predictions, references=true_labels)
	    return {
	        "precision": results["overall_precision"],
	        "recall": results["overall_recall"],
	        "f1": results["overall_f1"],
	        "accuracy": results["overall_accuracy"],
	    }


	# distill-bert model
	model = AutoModelForTokenClassification.from_pretrained(
	    "distilbert-base-uncased", num_labels=len(tagset), id2label=id2tag, label2id=tag2id
	)

	training_args = TrainingArguments(
	    output_dir="./checkpoints/Tag/bert",
	    learning_rate=2e-5,
	    per_device_train_batch_size=batch_size,
	    per_device_eval_batch_size=256,
	    num_train_epochs=10,
	    weight_decay=0.01,
	    evaluation_strategy="epoch",
	    save_strategy="epoch",
	    load_best_model_at_end=True,
	    push_to_hub=False,
	    save_total_limit=1,
	    metric_for_best_model="f1",
	    greater_is_better=True
	)

	trainer = Trainer(
	    model=model,
	    args=training_args,
	    train_dataset=ds_train_tokens,
	    eval_dataset=ds_dev_tokens,
	    tokenizer=tokenizer,
	    data_collator=data_collator,
	    compute_metrics=compute_metrics,
	    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
	)

	train_results = trainer.train()
	trainer.save_model()
	trainer.log_metrics("train", train_results.metrics)
	trainer.save_metrics("train", train_results.metrics)
	trainer.save_state()

	metrics = trainer.evaluate(ds_dev_tokens)
	trainer.log_metrics("eval", metrics)
	trainer.save_metrics("eval", metrics)
	print('----' * 32)
	print('FINAL')
	print()
	print(metrics)
	print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=256)
    parser.add_argument('-n', '--n_data', help='number of training data to subsample (None for full dataset)', type=int, default=None)
    parser.add_argument('-s', '--save', help='whether to save best model weights', type=bool, default=False)
    
    kwargs = vars(parser.parse_args())

    main(**kwargs)
