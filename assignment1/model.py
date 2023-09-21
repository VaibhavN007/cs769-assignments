import torch
import torch.nn as nn
import torch.nn.functional as F
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    words = vocab.word2id.keys()
    embedding_matrix = {}
    with open(emb_file, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            vector = np.asarray(split_line[1:], dtype='float32')
            embedding_matrix[word] = vector
    return embedding_matrix

class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)

        self.n_embed = len(self.vocab)
        self.d_embed = self.args.emb_size
        self.dropout = 0.3
        self.d_hidden = self.d_embed
        self.d_out = self.tag_size

        self.define_model_parameters()
        self.init_model_parameters()
        self.copy_embedding_from_numpy()

        # Use pre-trained word embeddings if emb_file exists
        if self.embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(self.embedding_matrix).float())
            self.embedding.weight.requires_grad = True

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        self.embedding = nn.Embedding(self.n_embed, self.d_embed)
        self.fc1 = nn.Linear(self.d_embed, self.d_hidden)
        self.bn1 = nn.BatchNorm1d(self.d_embed)
        self.fc2 = nn.Linear(self.d_hidden, self.d_hidden//2)
        self.bn2 = nn.BatchNorm1d(self.d_hidden//2)
        self.fc3 = nn.Linear(self.d_hidden//2, self.d_out)
        self.drop = nn.Dropout(self.dropout)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        torch.nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        self.embedding_matrix = np.copy(self.embedding.weight.data.numpy())
        embeddings_index = {}
        for i, word in self.vocab.id2word.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

    def forward(self, input):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        
        x = self.embedding(input.sents)

        embed_sum = (x * input.masks.unsqueeze(-1).float()).sum(dim=1)
        word_len = input.masks.sum(dim=1).unsqueeze(-1).float()
        x = embed_sum / word_len

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(self.drop(x))
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(self.drop(x))

        return x
