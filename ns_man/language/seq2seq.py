from ns_man.structs import *
from ns_man.language.nn.rnn import RNNEncoder, RNNDecoder

import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import json


class Seq2Seq(nn.Module):
    " Seq2Seq encoder-decoder for Program Synthesis "
    
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_id = self.decoder.pad_id
        self.start_id = self.decoder.start_id 
        self.end_id = self.decoder.end_id
        self.unk_id = self.decoder.unk_id
        self.max_len = self.decoder.max_length
        self.input_max_len = self.encoder.max_len

    def forward(self, src: Tensor, tgt: Tensor):
        encoder_outputs, encoder_hidden = self.encoder(src)
        decoder_outputs, decoder_hidden, attn = self.decoder(tgt, encoder_outputs, encoder_hidden)
        return decoder_outputs

    def forward_attn(self, src: Tensor, tgt: Tensor):
        encoder_outputs, encoder_hidden = self.encoder(src)
        decoder_outputs, decoder_hidden, attn = self.decoder(tgt, encoder_outputs, encoder_hidden)
        return decoder_outputs, attn
    
    # def forward_sample(self, src: Tensor):
    #     # works for single query
    #     assert len(src.shape) == 2 # Ls x D
    #     outputs = [self.start_id]
    #     for i in range(self.max_len):
    #         tgt = torch.tensor(outputs, dtype=longt, device=src.device).unsqueeze(0)
    #         out = self.forward(src.unsqueeze(0), tgt)
    #         token_idx = out.argmax(-1)[:, -1].item()
    #         outputs.append(token_idx)
    #         if token_idx == self.end_id:
    #             break
    #     return outputs

    def sample_output(self, src: Tensor):
        encoder_outputs, encoder_hidden = self.encoder(src)
        output_symbols, _, _ = self.decoder.forward_sample(encoder_outputs, encoder_hidden)
        return torch.stack(output_symbols).transpose(0,1)

    def sample_output_attn(self, src: Tensor):
        encoder_outputs, encoder_hidden = self.encoder(src)
        output_symbols, _, attns = self.decoder.forward_sample(encoder_outputs, encoder_hidden)
        return torch.stack(output_symbols).transpose(0,1), attns
    
    def reinforce_forward(self, src: Tensor):
        encoder_outputs, encoder_hidden = self.encoder(src)
        self.output_symbols, self.output_logprobs, _ = self.decoder.forward_sample(encoder_outputs, encoder_hidden, reinforce_sample=True)
        return torch.stack(self.output_symbols).transpose(0,1)

    def reinforce_backward(self, reward: float, entropy_factor=0.0):
        assert self.output_logprobs is not None and self.output_symbols is not None, 'must call reinforce_forward first'
        losses = []
        grad_output = []
        for i, symbol in enumerate(self.output_symbols):
            if len(self.output_symbols[0].shape) == 1:
                loss = - torch.diag(torch.index_select(self.output_logprobs[i], 1, symbol)).sum()*reward \
                       + entropy_factor*(self.output_logprobs[i]*torch.exp(self.output_logprobs[i])).sum()
            else:
                loss = - self.output_logprobs[i]*reward
            losses.append(loss.sum())
            grad_output.append(None)
        torch.autograd.backward(losses, grad_output, retain_graph=True)

    def make_collate_fn(self, device: str, pretrain: bool) -> Map[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]]:
        def _collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
            queries, programs = zip(*batch)
            queries  = pad_sequence(queries, padding_value=self.pad_id, batch_first=True) 
            programs = pad_sequence(programs, padding_value=self.pad_id, batch_first=True)
            return queries.to(device), programs.to(device)
        return _collate_fn

    def load(self, chp: str):
        self.load_state_dict(torch.load(chp))
        print(f'Loaded seq2seq weights from {chp}')


def make_seq2seq_net(cfg: Maybe[Union[Dict[str, Any], str]] = None, load=False) -> Seq2Seq:
    if cfg is None:
        cfg = json.load(open("./config/seq2seq_rnn_cfg.json"))
    else:
        cfg = json.load(open(cfg)) if isinstance(cfg, str) else cfg
    encoder = RNNEncoder(cfg)
    decoder = RNNDecoder(cfg)
    model = Seq2Seq(encoder, decoder)
    if cfg['seq2seq_checkpoint'] is not None and load:
        model.load(cfg['seq2seq_checkpoint'])
    return model