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
                 eos_token = "<EOS>"):
        if token_to_idx == None:
            token_to_idx = {}
        
        self._unk_token = unk_token
        self._sos_token = sos_token
        self._eos_token = eos_token
        self._unk_index = self.add_word(self._unk_token)
        self._sos_index = self.add_word(self._sos_token)
        self._eos_index = self.add_word(self._eos_token)

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
