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
from sklearn.model_selection import train_test_split

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
        
    def build_vocabulary(self, sentences, min_freq = 1):
        word_count = Counter()

        for sentence in sentences:
            tokens = sentence.lower().strip().split(" ")
            word_count.update(tokens)
        
        for word, count in word_count.items():
            if count >= min_freq:
                self.add_word(word)


test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
num_layers = 2 

class gru(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first = False):
        super(gru, self).__init__()
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
        
        self.gru = gru(input_size=embedding_size, hidden_size=rnn_hidden_size, batch_first=batch_first)
        self.fc1 = nn.Linear(in_features=rnn_hidden_size, out_features= rnn_hidden_size)
        self.fc2 = nn.Linear(in_features=rnn_hidden_size, out_features=num_classes)
    
    def forward(self, x_in, x_lengths, apply_softmax = False):
        x_embedded = self.emb(x_in)
        y_out = self.gru(x_embedded)

        if x_lengths is not None:
            y_out = column_gather(y_out, x_lengths)
        
        else:
            y_out = y_out[:,-1:]

        y_out = F.ReLU(self.fc1(F.dropout(y_out,0.5)))
        y_out = self.fc2(F.dropout(y_out,0.5))

        if apply_softmax:
            y_out = F.softmax(y_out, dim=1)
        
        return y_out
    
class SpookyDataset(Dataset):
    def __init__(self, train_df, test_df, vectorizer):
        self.train_df, self.val_df = train_test_split(train_df, test_size=0.2, random_state=42)
        self.test_df = test_df
        self.vectorizer = vectorizer

        self.train_size = len(self.train_df)
        self.val_size = len(self.val_df)
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size)
        }

        self.set_split('train')
       
    @classmethod
    def load_and_make_vectorizer(cls, train_csv, test_csv):
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")

        train_vocab = SequenceVocabulary()
        train_vocab.build_vocabulary(train_df.text.to_list())

        author_vocab = Vocabulary()
        author_vocab.add_words(sorted(train_df.author.unique()))

        vectorizer = TextVectorizer(train_vocab, author_vocab)
        return cls(train_df, test_df, vectorizer)
    
    def set_split(self, split="train"):
        self._target_split = split 
        self._target_df, self._target_size = self._lookup_dict[split]

    def __len__(self):
        return self._target_size
    
    def __getitem__(self, index):
        row = self._target_df.iloc[index]

        from_vector, length = self.vectorizer.vectorize(row.text)

        if self._target_split=='test':
            author_index = -1
        
        else:
            author_index = self.vectorizer.author_vocab.lookup_token(row.author)

        return {'x_data': from_vector,
                'y_target': author_index,
                'x_lengths': length}
    
    def num_batches(self, batch_size):
        return len(self)//batch_size

args = Namespace(
    train_csv="train.csv",
    test_csv="test.csv",
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="model_storage",

    embedding_size=128,
    rnn_hidden_size=128,
    num_layers=2, 
    bidirectional=False,
    

    seed=9248,
    learning_rate=0.001,
    batch_size=64,
    num_epochs=100,
    early_stopping_criteria=5,

    cuda=True,   
    reload_from_file=False,
    expand_filepaths_to_save_dir=True
)

def make_train_state(args):
    return{
        'stop_early': False,
        'early_stopping_step': 0,
        'early_stopping_best_val': 1e8,
        'learning_rate': args.learning_rate,
        'epoch_index': 0,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': -1,
        'test_acc': -1,
        'model_filename': args.model_state_file
    }

def compute_accuracy(y_pred, y_target):
    _, y_pred_indices = y_pred.max(dim=1)

    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100

dataset = SpookyDataset.load_dataset_and_make_vectorizer(args.train_csv, args.test_csv)
vectorizer = dataset.vectorizer
model = AuthorClassifier(
    num_embeddings=len(vectorizer.text_vocab), 
    embedding_size=args.embedding_size, 
    rnn_hidden_size=args.rnn_hidden_size,
    num_classes=len(vectorizer.author_vocab),
    batch_first=True,
    padding_idx=vectorizer.text_vocab._mask_index 
)

device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
args.device = device
model = model.to(device)
loss_func = nn.CrossEntropyLoss(weight=dataset.class_weights.to(args.device))
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

train_state = make_train_state(args)