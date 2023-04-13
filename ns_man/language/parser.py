from ns_man.structs import *
from ns_man.language.word_embedding import make_word_embedder
from ns_man.language.concept_tagger import make_concept_tagger_rnn, TAGSET
from ns_man.language.seq2seq import make_seq2seq_net

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, AutoTokenizer
import json
import numpy as np
from scipy.optimize import linear_sum_assignment


class LanguageParser:

    iob_to_tag = {'B-LOC' : '<L>', 'B-COL' : '<C>', 'B-REL' : '<R>', 'B-HREL' : '<H>', 'B-MAT' : '<M>', 'B-CAT' : '<Y>',}
    
    concept_args_map = {'filter_category' : '<Y>', 'filter_unique': '<Y>', 'filter_color': '<C>', 'filter_material': '<M>',
        'locate': '<L>', 'relate': '<R>', 'hyper_relate': '<H>'}

    def __init__(self, cfg: Union[str, Dict[str,Any]]):
        self.cfg = json.load(open(cfg)) if isinstance(cfg, str) else cfg
        self.device = self.cfg["device"]
        self.tagset = TAGSET
        self.id2tag = {k:v for k,v in enumerate(sorted(self.tagset))}
        self.tag2id = {v:k for k,v in self.id2tag.items()}
        self.use_bert_tagger = self.cfg["use_bert_tagger"]
        if not self.use_bert_tagger:
            self.word_embedder = make_word_embedder()
            self.tagger = make_concept_tagger_rnn(self.cfg['tagger_rnn_cfg'], load=True).to(self.device)
            self.tagger.eval()
            self.tag = self._tag_rnn
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.tagger = AutoModelForTokenClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=len(self.tagset), id2label=self.id2tag, label2id=self.tag2id
            ).eval().to(self.device)
            self.tagger.load_state_dict(torch.load(self.cfg['bert_checkpoint']))
            self.tag = self._tag_bert
        self.seq2seq = make_seq2seq_net(load=True).eval().to(self.device)
        self.vocabs = json.load(open(self.cfg["vocabs_path"]))
        self.vocabs['id2word'] = {v:k for k,v in self.vocabs['word2id'].items()}
        self.vocabs['id2prog'] = {v:k for k,v in self.vocabs['prog2id'].items()}
        self.dummy_input = torch.as_tensor([self.seq2seq.pad_id for _ in range(self.seq2seq.input_max_len)],
            dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _tag_rnn(self, text_input: Union[str, List[str]]) -> List[str]:
        single = False
        if isinstance(text_input, str):
            # single text input, make batch
            text_input = [text_input]
            single = True
        text_input = list(map(self.tokenize_query, text_input))
        text_vectors = list(map(self.word_embedder, [" ".join(ts) for ts in text_input]))
        text_lens = [v.shape[0] for v in text_vectors]
        text_vectors = pad_sequence(text_vectors,
            batch_first=True, padding_value=self.tagger.pad_id).to(self.device)
        tag_output = self.tagger(text_vectors)[0].argmax(-1).cpu().tolist()
        iob_tags = [[self.id2tag[t] for t in tags][:l] for l, tags in zip(text_lens, tag_output)]
        compressed_tags, tagged_sequence, concept_map = zip(*[self.compress_iob_tags(t, q) 
            for t,q in zip(iob_tags, text_input)])
        iob_tags = iob_tags if not single else iob_tags[0]
        compressed_tags = compressed_tags if not single else compressed_tags[0]
        tagged_sequence = tagged_sequence if not single else tagged_sequence[0]
        return iob_tags, compressed_tags, tagged_sequence, concept_map

    @torch.no_grad()
    def _tag_bert(self, text_input: Union[str, List[str]]) -> List[str]:
        single = False
        if isinstance(text_input, str):
            # single text input, make batch
            text_input = [text_input]
            single = True
        text_tokens = list(map(self.tokenizer.tokenize, text_input))
        text_lens = [len(q) for q in text_tokens]
        text_input = self.tokenizer(text_input, truncation=True, padding=True,
            return_tensors="pt")
        logits = self.tagger(**{k:v.to(self.device) for k,v in text_input.items()}).logits
        tag_output = logits.argmax(-1).cpu().tolist()
        iob_tags = [[self.id2tag[t] for t in tags[1:-1]][:l] for l, tags in zip(text_lens, tag_output)]
        compressed_tags, tagged_replaced, concept_map = zip(*[self.compress_iob_tags(t, q) 
            for t,q in zip(iob_tags, text_tokens)])
        iob_tags = iob_tags if not single else iob_tags[0]
        compressed_tags = compressed_tags if not single else compressed_tags[0]
        tagged_replaced = tagged_replaced if not single else tagged_replaced[0]
        concept_map = concept_map if not single else concept_map[0]
        return iob_tags, compressed_tags, tagged_replaced, concept_map

    @torch.no_grad()
    def _generate_program_tokens(self, tagged_input: List[str]):
        tagged_input_ids = self.encode_query([" ".join(ts) for ts in tagged_input])
        tagged_input_padded = pad_sequence([self.dummy_input] + tagged_input_ids, batch_first=True, padding_value=self.seq2seq.pad_id)
        program_ids, _ = self.seq2seq.sample_output_attn(tagged_input_padded)
        program_ids = program_ids[1:].cpu().tolist() # remove dummy
        untils = [p.index(self.seq2seq.end_id) if self.seq2seq.end_id in p else len(p) for p in program_ids]
        program_tokens = [p[:until][::-1] for p, until in zip(program_ids, untils)] # re-reverse
        program_tokens = [[self.vocabs['id2prog'][x] for x in p] for p in program_tokens]
        return program_tokens

    @torch.no_grad()
    def _generate_program(self, 
                        tagged_input: List[str],
                        tagged_replaced: List[str],
                        concept_map: List[Dict[str, Any]]
    ):
        tagged_input_ids = self.encode_query([" ".join(ts) for ts in tagged_input])
        tagged_input_padded = pad_sequence([self.dummy_input] + tagged_input_ids, batch_first=True, padding_value=self.seq2seq.pad_id)
        program_ids, attention_scores = self.seq2seq.sample_output_attn(tagged_input_padded)
        program_ids = program_ids[1:].cpu().tolist() # remove dummy
        attention_scores = torch.stack([x[1:].squeeze(1) for x in attention_scores], dim=1) # batch first, remove dummy
        untils = [p.index(self.seq2seq.end_id) if self.seq2seq.end_id in p else len(p) for p in program_ids]
        program_tokens = [p[:until][::-1] for p, until in zip(program_ids, untils)] # re-reverse
        program_tokens = [[self.vocabs['id2prog'][x] for x in p] for p in program_tokens]
        #programs_full = []
        programs_with_args = []
        # restore arguments from tag-masked attention scores
        for p, scores, ts, tsr, cm in zip(program_tokens, attention_scores, tagged_input, tagged_replaced, concept_map):
            scores = scores[:len(p), :len(ts)].flip([0,])   # re-reverse
            #pfull = []
            p_w_args = []
            for step in range(len(p)):
                function = p[step]
                if function not in self.concept_args_map.keys():
                    #pfull.append(function)
                    p_w_args.append({'function': function, 'arguments': None, 'scores': None})
                    continue
                tag = self.concept_args_map[function]
                # mask attention scores according to tag
                mask = torch.as_tensor([1 if token==tag else 0 for token in ts], dtype=bool, device=self.device)
                tag_masked_scores = torch.where(mask, scores[step], torch.zeros_like(scores[step]))
                #argument = cm[tsr[tag_masked_scores.argmax().item()]]
                #pfull.append(f"{function}[{argument}]")
                _values, _ids = [t.tolist() for t in torch.sort(tag_masked_scores, descending=True)]
                _values, _ids = zip(*[(v,i) for v,i in zip(_values, _ids) if v>0])
                _args = [cm[tsr[i]] for i in _ids]
                p_w_args.append({'function': function, 'arguments': _args, 'scores': _values})   
            #programs_full.append(pfull)
            programs_with_args.append(p_w_args)
        #return programs_full
            programs_retrieved = list(map(self.recover_arguments, programs_with_args))
        return programs_retrieved
        
    def parse(self, text_input: Union[str, List[str]]):
        single = False
        if isinstance(text_input, str):
            # single text input, make batch
            text_input = [text_input]
            single = True
        # first tag the sequence
        _, tagged_input, tagged_replaced, concept_map = self.tag(text_input)
        programs = self._generate_program(tagged_input, tagged_replaced, concept_map)
        programs = programs[0] if single else programs
        return programs

    def compress_iob_tags(self, tags: List[str], query_tokens: str) -> str:
        assert len(query_tokens) == len(tags)
        compressed, replaced, concept_map = [], [], {}
        counter = {v: 0 for v in list(self.iob_to_tag.values())}
        current = None
        for t, x in zip(tags, query_tokens):
            if x.startswith('##') or x in ['-']:
                continue
            if t in self.iob_to_tag.keys():
                current = self.iob_to_tag[t]
                compressed.append(current)
                current_cnt = current[:-1] + str(counter[current]) + '>'
                replaced.append(current_cnt)
                counter[current] += 1
                concept_map[current_cnt] = [x]
            elif t.startswith('I-') and current is not None:
                if self.iob_to_tag['B-'+t.split('-')[1]] == current:
                    concept_map[current_cnt].append(x)
                continue
            else:
                compressed.append(x)
                replaced.append(x)
        concept_map = {k: " ".join(v) for k, v in concept_map.items()}
        return compressed, replaced, concept_map

    @staticmethod
    def tokenize_query(query: str) -> str:
        q = query + '?' if query[-1] != '?' else query
        return q.replace('?',' ?').replace(';', ' ;').replace(',', ' ,').replace("'s", " is").replace("Whats", "What is").split()

    def encode_query(self, text_input: Union[str, List[str]]) -> Tensor:
        if isinstance(text_input, str):
            # single text input, make batch
            text_input = [text_input]
        text_tokens = list(map(self.tokenize_query, text_input))
        text_ids = [
            torch.as_tensor(
                [self.vocabs['word2id']["<START>"]]
                + [
                    self.vocabs['word2id'][t] if t in self.vocabs['word2id'].keys() else self.vocabs['word2id']["<UNK>"]
                    for t in ts
                ]
                + [self.vocabs['word2id']["<END>"]],
                dtype=torch.long, device=self.device
            )
            for ts in text_tokens
        ]
        return text_ids

    # @staticmethod
    # def get_best_arguments(data: List[Dict[str, Any]]) -> List[str]:
    #     output = []
    #     used_args = {}c[0]
    #     for item in data:
    #         func = item['function']
    #         args = item['arguments']
    #         scores = item['scores']
    #         if args is None:
    #             output.append(func)
    #         else:
    #             double_arg = True if len(set(args)) < len(args) else False
    #             arg_score_pairs = list(zip(args, scores))
    #             arg_score_pairs.sort(key=lambda x: x[1], reverse=True)
    #             for arg, score in arg_score_pairs:
    #                 if arg not in used_args:
    #                     used_args.add(arg)
    #                     output.append(f"{func}[{arg}]")
    #                     break
    #                 elif arg in used_args and double_arg:
    #                 	output.append(f"{func}[{arg}]")
    #                 	break
    #     return output

    @staticmethod
    def recover_arguments(data: List[Dict[str, Any]]) -> List[str]:
        # hungarian matching algorithm for linear sum assignment
        cost_maps = {}
        args = {}
        for item in data:
            if item['arguments'] is not None:
                arg_score_pairs = list(zip(item['arguments'], item['scores']))
                arg_score_pairs.sort(key=lambda x: x[0])
                if item['function'] not in cost_maps.keys():
                    cost_maps[item['function']] = [[x[1] for x in arg_score_pairs]]
                    args[item['function']] = [x[0] for x in arg_score_pairs]
                else:
                    cost_maps[item['function']].append([x[1] for x in arg_score_pairs])
        # normalize row-wise to compare accross rows
        cost_maps = {k: - np.asarray(v) / np.asarray(v).sum(axis=1)[:, np.newaxis] for k,v in cost_maps.items()}
        assignments = {k: [args[k][i] for i in linear_sum_assignment(v)[1].tolist()] for k,v in cost_maps.items()}
        output = []
        used_args = set()
        for item in data:
            if item['arguments'] is None:
                output.append(item['function'])
            else:
                try:
                    arg = assignments[item['function']].pop(0)
                    used_args.add(arg)
                except IndexError:
                    arg = "bowl"
                output.append(f"{item['function']}[{arg}]")
        return output