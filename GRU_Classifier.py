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
    def __init__(self, token_to_idx=None, unk_token = "<UNK>", sos_token = "<SOS>", 
                 eos_token = "<EOS>", mask_token = "<MASK>"):
        if token_to_idx == None:
            token_to_idx = {}
        
        self._unk_token = unk_token
        self._sos_token = sos_token
        self._eos_token = eos_token
        self._mask_token = mask_token
        self._unk_index = self.add_word(self._unk_token)
        self._sos_index = self.add_word(self._sos_token)
        self._eos_index = self.add_word(self._eos_token)
        self._mask_index = self.add_word(self._mask_token)

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
    
    def build_vocabulary(self, sentences):
        for sentence in sentences:
            tokens = sentence.lower().strip().split(" ")
            full_sequence = [self._sos_token] + tokens + [self._eos_token]
            self.add_words(full_sequence)

    def lookup_token(self, token):
        return self._token_to_idx.get(token, self._unk_index)
    
    def lookup_index(self, index):
        return self._idx_to_token.get(index, self._unk_token)
    
    def __len__(self):
        return len(self._token_to_idx)
    

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
        out_vector[len(indices):] = self.char_vocab.mask_index

        return out_vector, len(indices)

    def to_serializable(self):
        return {
            'vocabulary': self.vocabulary.to_serializable(),
        }

    @classmethod
    def from_serializable(cls, contents):
        vocab = Vocabulary.from_serializable(contents['vocabulary'])
        return cls(vocabulary=vocab)

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
num_layers = 2 

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first = False):
        super(GRU, self).__init__()
        self.rnn = nn.GRUCell(input_size, hidden_size)

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        def _intial_hidden(self, batch_size):
            return torch.zeros((batch_size, hidden_size))
        
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

            for t in range(seq_size):
                hidden_t = self.rnn(x_in[t], hidden_t)
                hiddens.append(hidden_t)

            hiddens = torch.stack(hiddens)

            if self.batch_first:
                hiddens = hiddens.permute(1,0,2)
            
            return hiddens
