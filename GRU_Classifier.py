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
            self._token_to_idx[index] = token
            self._idx_to_token[token] = index
        return token 
    
    def add_words(self, tokens):
        return [self.add_token(token) for token in tokens]
    
    def build_vocabulary(self, sentences):
        for sentence in sentences:
            current = []
            sentence.strip.split(" ")
            self.add_word(sentence)

    def lookup_token(self, token):
        self._token_to_idx[token]

    def lookup_index(self, index):
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]
    
    def __len__(self):
        return len(self._token_to_idx)
