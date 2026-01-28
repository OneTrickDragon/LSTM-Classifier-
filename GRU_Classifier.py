import os
from argparse import Namespace
from collections import Counter
import json
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Vocabulary(object):
    def __init__(self, token_to_idx=None):
        if token_to_idx == None:
            token_to_idx = {}

        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token
                             for token, idx in self.token_to_idx.items()}
        
    def add_word(self, token):
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index
    
    def add_words(self, tokens):
        return [self.add_word(token) for token in tokens]

    def lookup_token(self, token):
        return self._token_to_idx[token]
    
    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]
    
    def __len__(self):
        return len(self._token_to_idx)

    def to_serializable(self):
        return {'token_to_idx': self._token_to_idx}
    
    @classmethod 
    def from_serializable(cls, contents):
        return cls(**contents)

class TextVectorizer(object):
    def __init__(self, text_vocab, author_vocab):
        self.text_vocab = text_vocab
        self.author_vocab = author_vocab

    def vectorize(self, text, vector_length = -1):
        indices = [self.text_vocab._sos_index]
        indices.extend(self.text_vocab.lookup_token(token) for token in text)
        indices.append(self.text_vocab._eos_index)

        if vector_length <= 0:
            vector_length = len(indices)

        out_vector = np.zeros(vector_length, dtype=np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.text_vocab.mask_index

        return out_vector, len(indices)

    def to_serializable(self):
        return {
            'text_vocabulary': self.text_vocab.to_serializable(),
            'author_vocabulary': self.author_vocab.to_serializable(),
        }

    @classmethod
    def from_serializable(cls, contents):
        text_vocab = Vocabulary.from_serializable(contents['text_vocab'])
        author_vocab =  Vocabulary.from_serializable(contents['author_vocab'])
        return cls(text_vocab=text_vocab, author_vocab=author_vocab)


class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", sos_token="<SOS>",
                 eos_token="<EOS>"):
        
        super(SequenceVocabulary, self).__init__(token_to_idx)
        self._unk_token = unk_token
        self._sos_token = sos_token
        self._eos_token = eos_token
        self._mask_token = mask_token
        self._unk_index = self.add_word(self._unk_token)
        self._sos_index = self.add_word(self._sos_token)
        self._eos_index = self.add_word(self._eos_token)
        self._mask_index = self.add_word(self._mask_token)
    
    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self._unk_token,
                         'mask_token': self._mask_token,
                         'sos_token': self._sos_token,
                         'eos_token': self._eos_token})
        return contents
    
    def lookup_token(self, token):
        if self._unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
num_layers = 2 

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first = False):
        super(GRU, self).__init__()
        self.rnn = nn.GRUCell(input_size, hidden_size)

        self.hidden_size = hidden_size
        self.batch_first = batch_first

    def _intial_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size))
    
    def forward(self, x_in, initial_hidden = None):
        if self.batch_first:
            batch_size, seq_size, feat_size = x_in.size()
            x_in = x_in.permute(1,0,2)
        else:
            seq_size, batch_size, feat_size = x_in.size()

        hiddens = []
        if initial_hidden is None:
            initial_hidden = self._intial_hidden(batch_size)
            initial_hidden = initial_hidden.to(x_in.device)

        hidden_t = initial_hidden
        for t in range(seq_size):
            hidden_t = self.rnn(x_in[t], hidden_t)
            hiddens.append(hidden_t)

        hiddens = torch.stack(hiddens)

        if self.batch_first:
            hiddens = hiddens.permute(1,0,2)
        
        return hiddens

def column_gather(y_out, x_lengths):
    x_lengths = x_lengths.long().detach().cpu().numpy() - 1

    out = []
    for batch_index, column_index in enumerate(x_lengths):
        out.append(y_out[batch_index, column_index])

    return torch.stack(out)

class AuthorClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_size, num_classes, rnn_hidden_size, 
                 batch_first = True, padding_idx=0):
        super(AuthorClassifier, self).__init__()

        self.emb = nn.Embedding(num_embeddings=num_embeddings,
                                embedding_dim=embedding_size,
                                padding_idx=padding_idx)
        
        self.GRU = GRU(input_size=embedding_size, hidden_size=rnn_hidden_size, batch_first=batch_first)
        self.fc1 = nn.Linear(in_features=rnn_hidden_size, out_features= rnn_hidden_size)
        self.fc2 = nn.Linear(in_features=rnn_hidden_size, out_features=num_classes)
    
    def forward(self, x_in, x_lengths, apply_softmax = False):
        x_embedded = self.emb(x_in)
        y_out = self.GRU(x_embedded)

        if x_lengths is not None:
            y_out = column_gather(y_out, x_lengths)
        
        else:
            y_out = y_out[:,-1:]

        y_out = F.ReLU(self.fc1(F.dropout(y_out,0.5)))
        y_out = self.fc2(F.dropout(y_out,0.5))

        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)
        
        return y_out