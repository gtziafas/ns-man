from ns_man.structs import *
import torch
from torch.nn.utils.rnn import pad_sequence

Node = Dict[str, Any]
Program = List[Node]
Tokens = List[str]
Vocabulary = Dict[str, int]

SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}


def _tokenize_programs(programs: List[Program]) -> List[Tokens]:
    return [ 
        [
             node["type"]
             + ("" if not node["value_inputs"] else "[" + node["value_inputs"][0] + "]")
             for node in p
         ]
         for p in programs
    ]


def postprocess_programs(programs: List[Tokens], tokenize: bool, reverse: bool) -> List[Tokens]:
    # data gen output breaks chain order for "same_" fn in some samples
    # manual fix
    programs = programs if not tokenize else _tokenize_programs(programs)
    query_fns = ['query_color', 'query_category', 'query_material']
    equal_fns = ['equal_color', 'equal_category', 'equal_material']
    for i, p in enumerate(programs):
        if p[-1] in equal_fns and (p[-2] == p[-3] and p[-2] in query_fns):
            swap_id = p[1:].index('scene') + 1 
            programs[i] = p[:swap_id] + [p[-3]] + p[swap_id:-3] + p[-2:]
        if reverse:
            programs[i] = programs[i][::-1]
    # replace filter_unique with filter_cat + unique
    for i, p in enumerate(programs):
        for j, node in enumerate(p):
            if node.startswith('filter_unique'):
                _rest = '' if node == 'filter_unique' else node[len('filter_unique'):]
                programs[i][j] = f'filter_category{_rest}'
                put_at = j if reverse else j+1
                programs[i][put_at:put_at] = ['unique'] 
    return programs


def tokenize_programs(programs: List[Program]) -> Tuple[List[Tokens], Vocabulary]:
    tokens = postprocess_programs(programs, tokenize=True, reverse=True)
    
    all_tokens = set({})
    for t in tokens:
        all_tokens = all_tokens.union(set(t))

    prog_vocab = {
        **SPECIAL_TOKENS,
        **{v: k + len(SPECIAL_TOKENS) for k, v in enumerate(sorted(all_tokens))},
    }
    return tokens, prog_vocab


def embed_programs(programs: List[Program], max_len: Maybe[int] = None) -> Tuple[Tensor, Vocabulary]:
    tokens, prog_vocab = tokenize_programs(programs)
    token_ids = [
            torch.tensor(
                [prog_vocab["<START>"]]
                + [
                    prog_vocab[t] if t in prog_vocab.keys() else prog_vocab["<UNK>"]
                    for t in ts
                ]
                + [prog_vocab["<END>"]],
                dtype=longt,
            )
            for ts in tokens
        ]
    if max_len is not None:
        token_ids = pad_sequence(token_ids + [torch.empty(max_len, dtype=longt)],
            padding_value=SPECIAL_TOKENS["<PAD>"], batch_first=True)
        assert token_ids.shape[0] == len(programs)
    return token_ids, {v:k for k,v in prog_vocab.items()}
