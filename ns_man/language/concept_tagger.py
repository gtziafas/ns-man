from ns_man.structs import *
from ns_man.language.nn.rnn import RNNEncoder

import json
import evaluate
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


TAGSET = [
    '0',
    'B-CAT',
    'B-COL',
    'B-HREL',
    'B-LOC',
    'B-MAT',
    'B-REL',
    'I-CAT',
    'I-COL',
    'I-HREL',
    'I-LOC',
    'I-MAT',
    'I-REL'
 ]


class ConceptTagger(nn.Module):
  def __init__(self, cfg: Union[str, Dict[str, Any]]):
      super().__init__()
      self.cfg = json.load(open(cfg)) if isinstance(cfg, str) else cfg
      self.tagset = TAGSET
      self.tag_vocab_size = len(self.tagset)
      self.bidirectional = self.cfg['bidirectional_encoder']
      self.use_pretrain_embeddings = self.cfg["use_pretrain_embeddings"]
      self.query_vocab_size = self.cfg["query_vocab_size"]
      self.encoder = RNNEncoder(self.cfg)
      fc_input_dim = 2 * self.cfg["hidden_size"] if self.bidirectional else self.cfg["hidden_size"]
      self.fc = nn.Linear(fc_input_dim, self.tag_vocab_size)
      self.fc_dropout = nn.Dropout(self.cfg["fc_dropout"])
      self.seqeval = evaluate.load("seqeval")
      self.pad_id = self.cfg["special_tokens"]["pad"]
      # if self.cfg['load_checkpoint'] is not None:
      # 	self.load()

  def forward(self, queries: Tensor) -> Tensor:
      word_states, phrase_context = self.encoder(queries)
      word_states = self.fc_dropout(word_states)
      word_tags = self.fc(word_states)
      return word_tags, phrase_context

  @torch.no_grad()
  def compute_metrics(self, predictions: Tensor, truth: Tensor, ignore_idx: int = -1) -> Dict[str, float]:
        B, N, K = predictions.shape
        # Removing padding (-1) from the predictions tensor
        predictions = predictions.reshape(-1, K)
        truth = truth.reshape(-1, K)
        mask = (truth != ignore_idx).all(axis=1)
        predictions = predictions[mask]
        truth = truth[mask]

        # Calculating the scores
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(truth, axis=1)

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='micro', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='micro', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)
        
        return {'acc': accuracy, 'p': precision, 'r': recall, 'f1': f1}

  @torch.no_grad()
  def compute_metrics_2(self, predictions, labels, ignore_idx: int = -1):
        labels = labels.argmax(2).cpu().tolist()
        predictions = predictions.argmax(2).cpu().tolist()

        true_predictions = [
            [self.tagset[p] for (p, l) in zip(prediction, label) if l != ignore_idx]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.tagset[l] for (p, l) in zip(prediction, label) if l != ignore_idx]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "p": results["overall_precision"],
            "r": results["overall_recall"],
            "f1": results["overall_f1"],
            "acc": results["overall_accuracy"],
        }

  def make_collate_fn(self, device: str, pretrain: bool) -> Map[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]:
      def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
          words, tags = zip(*batch)
          words = pad_sequence(words, batch_first=True, padding_value=self.pad_id)
          if not pretrain:
            words = words.long()
          tags = pad_sequence(tags, batch_first=True, padding_value=self.pad_id)
          return words.to(device), tags.to(device)
      return collate_fn

  def load(self):
  	assert self.cfg["load_checkpoint"] is not None, "checkpoint is empty in config json"
  	self.load_state_dict(torch.load(self.cfg["load_checkpoint"]))
  	print(f'Loaded tagger weights from {self.cfg["load_checkpoint"]}')


def make_concept_tagger_rnn(cfg: Maybe[str] = None, load: bool=False) -> ConceptTagger:
  cfg = cfg or "./config/tagger_rnn_cfg.json"
  model = ConceptTagger(cfg)
  if load:
    model.load()
  return model
