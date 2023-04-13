from ns_man.structs import *
import torch


# Glove word embedding
WordEmbedder = Map[List[str], Tensor]

# pre-trained GloVe embeddings with glove_dim=96
def glove_embeddings(version: str = 'sm', remove_punct: bool = True) -> WordEmbedder:
    import spacy
    _glove = spacy.load(f'en_core_web_{version}') 
    print('Loaded spacy word embeddings...')
    def embedd(sent: List[str]) -> Tensor:
        sent = sent.replace("-","")
        sent_proc = _glove(sent)
        #vectors = [word.vector for word in sent_proc if word.pos_ !='PUNCT' or not remove_punct]
        vectors = [word.vector for word in sent_proc]
        return torch.stack([torch.tensor(v, dtype=floatt) for v in vectors])
    def embedd_many(sents: List[str]) -> List[Tensor]:
        return list(map(embedd, sents))
    return embedd


# make word embedder function
def make_word_embedder(embeddings: str = 'glove_sm', **kwargs) -> WordEmbedder:
    if embeddings.startswith('glove_'):
        version = embeddings.split('_')[1]
        if version not in ['sm', 'md', 'lg']:
            raise ValueError('See utils/embeddings.py for valid embedding options')
        embedder = glove_embeddings(version)
    else:
        raise ValueError('See utils/embeddings.py for valid embedding options')
    return embedder