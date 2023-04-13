from ns_man.structs import *

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable


class AttentionLayer(nn.Module):
    " Attention layer between RNN encoder and decoder"
    def __init__(self, dim: int, use_weight: bool=False, hidden_size: int=512):
        super().__init__()
        self.use_weight = use_weight
        self.hidden_size = hidden_size
        if use_weight:
            # use weighted attention 
            self.attn_weight = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out = nn.Linear(2 * dim, dim)

    def forward(self, outputs: Tensor, context: Tensor) -> Tensors:
        """
        - args
        tgt : Tensor
            decoder output, B x Lt x D 
        context : Tensor
            context vector from encoder, B x Ls x D
        - returns
        output : Tensor
            attention layer output, B x Lt x D 
        attn : Tensor
            attention map, B x Lt x Ls
        """
        batch_size = outputs.size(0)
        hidden_size = outputs.size(2)
        src_len = context.size(1)
        
        if self.use_weight:
            outputs = self.attn_weight(outputs.contiguous().view(-1, 
                hidden_size)).view(batch_size, -1, hidden_size)

        attn = torch.bmm(outputs, context.transpose(1, 2))
        attn = F.softmax(attn.view(-1, src_len), dim=1)
        attn = attn.view(batch_size, -1, src_len)   # B x Ls x Lt

        scores = torch.bmm(attn, context) # B x Lt x D
        catted = torch.cat([scores, outputs], dim=2)    # B x Lt x 2*D  
        output = self.out(catted.view(-1, 2*hidden_size)).tanh()
        output = output.view(batch_size, -1, hidden_size)

        return output, attn


class BaseRNN(nn.Module):
    " Abstract RNN module "
    
    def __init__(self, 
                 vocab_size: int, 
                 max_len: int, 
                 hidden_size: float, 
                 input_dropout: float, 
                 rnn_dropout: float, 
                 num_layers: int,
                 bidirectional: bool, 
                 rnn_cell: str = 'gru'
                ):
        super(BaseRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_dropout = rnn_dropout
        self.bidirectional = bidirectional
        self.input_dropout = nn.Dropout(input_dropout)
        try:
            self.rnn_cell = getattr(nn, rnn_cell.upper())
        except AttributeError:
            raise ValueError(f'Unsupported RNN cell type {rnn_cell}')

    def make_rnn(self, embedding_size: int):
        return self.rnn_cell(embedding_size, self.hidden_size, self.num_layers,
            dropout=self.rnn_dropout, bidirectional=self.bidirectional, batch_first=True)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class RNNEncoder(BaseRNN):
    " RNN Encoder:  Query -> (Hidden, Context) "
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config['query_vocab_size'], 
                         config['query_max_len'], 
                         config['hidden_size'], 
                         config['input_dropout'], 
                         config['rnn_dropout'], 
                         config['num_encoder_layers'], 
                         config['bidirectional_encoder'],
                         config['rnn_cell']
                        )
        self.cfg = config
        self.pad_id = self.cfg['special_tokens']['pad']

        # Create source embedding layer if desired
        if not self.cfg['use_pretrain_embeddings']:
            assert self.cfg['query_vocab_size'] is not None
            self.embedding = nn.Embedding(self.cfg['query_vocab_size'], 
                                          self.cfg['embedding_size']
                                        )
        else:
            self.embedding = nn.Identity()

        self.rnn = self.make_rnn(config['embedding_size'])

    def forward(self, src: Tensor) -> Tensors:
        # assert we are properly allowing pretrained word embeddings
        if len(src.shape) == 3:
            batch_size, src_len, input_size = src.shape 
            assert input_size == self.cfg['embedding_size'] and self.cfg['use_pretrain_embeddings']
        elif len(src.shape) == 2:
            batch_size, src_len = src.shape
            assert not self.cfg['use_pretrain_embeddings']
        embeds = self.embedding(src)
        embeds = self.input_dropout(embeds)
        outputs, hidden = self.rnn(embeds) 
        return outputs, hidden


class RNNDecoder(BaseRNN):
    " RNN Decoder: (Hidden, Context) -> Program "
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config['program_vocab_size'], 
                         config['program_max_len'], 
                         config['hidden_size'], 
                         config['input_dropout'], 
                         config['rnn_dropout'], 
                         config['num_decoder_layers'], 
                         config['bidirectional_decoder'],
                         config['rnn_cell']
                        )
        self.cfg = config
        self.max_length = self.cfg['program_max_len']
        self.output_size = self.cfg['program_vocab_size']
        self.bidirectional = self.cfg['bidirectional_decoder']
        self.bidirectional_encoder = self.cfg['bidirectional_encoder']
        self.hidden_size = self.cfg['hidden_size'] if not self.bidirectional_encoder else 2*self.cfg['hidden_size']
        self.embedding_size = self.cfg['embedding_size']
        self.use_attention = self.cfg['use_attention']
        self.pad_id = self.cfg['special_tokens']['pad']
        self.start_id = self.cfg['special_tokens']['start']
        self.end_id = self.cfg['special_tokens']['end']
        self.unk_id = self.cfg['special_tokens']['unk']

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.rnn = self.make_rnn(self.embedding_size)
        self.out_linear = nn.Linear(self.hidden_size, self.output_size)
        if self.use_attention:
            self.attention = AttentionLayer(self.hidden_size)

    def forward_step(self, tgt: Tensor, hidden: Tensor, encoder_outputs: Tensor) -> Tensors:
        batch_size, output_size = tgt.shape

        embedded = self.embedding(tgt)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        output = self.out_linear(output.contiguous().view(-1, self.hidden_size))
        #predicted_softmax = F.log_softmax(output.view(batch_size, output_size, -1), 2)
        output = output.view(batch_size, output_size, -1)

        return output, hidden, attn

    def forward(self, tgt: Tensor, encoder_outputs: Tensor, encoder_hidden: Tensor) -> Tensor:
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs, decoder_hidden, attn = self.forward_step(tgt, decoder_hidden, encoder_outputs)

        return decoder_outputs, decoder_hidden, attn

    def forward_sample(self, encoder_outputs: Tensor, encoder_hidden: Tensor, reinforce_sample: bool=False):
        device = encoder_outputs.device
        batch_size = encoder_hidden[0].shape[1] if isinstance(encoder_hidden, tuple) else encoder_hidden.shape[1]
        decoder_hidden = self._init_state(encoder_hidden)    
        decoder_input = Variable(torch.LongTensor(batch_size, 1).fill_(self.start_id).to(device))
        
        output_logprobs = []
        output_symbols = []
        output_lengths = np.array([self.max_length] * batch_size)

        def decode(i: int, output: Tensor, reinforce_sample=reinforce_sample):
            output_logprobs.append(output.squeeze())
            if reinforce_sample:
                dist = torch.distributions.Categorical(probs=torch.exp(output.view(batch_size, -1))) # better initialize with logits
                symbols = dist.sample().unsqueeze(1)
            else:
                symbols = output.topk(1)[1].view(batch_size, -1)
            output_symbols.append(symbols.squeeze(1))

            eos_batches = symbols.data.eq(self.end_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((output_lengths > i) & eos_batches) != 0
                output_lengths[update_idx] = len(output_symbols)

            return symbols

        attns = []
        for i in range(self.max_length):
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_logprobs = F.log_softmax(decoder_output, 2)
            decoder_input = decode(i, decoder_logprobs)
            attns.append(step_attn)

        return output_symbols, output_logprobs, attns

    def _init_state(self, encoder_hidden: Any) -> Tensor:
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h: Tensor) -> Tensor:
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h