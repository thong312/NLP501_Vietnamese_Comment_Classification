from collections import Counter
import re
import torch
from torch import nn
from vihsd.data import normalize

PAD, UNK = 0, 1


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", normalize(text).lower(), flags=re.UNICODE)


def build_vocab(texts, limit=30000):
    counter = Counter(token for text in texts for token in tokenize(text))
    words = sorted(counter, key=lambda word: (-counter[word], word))[:limit - 2]
    return {"<PAD>": PAD, "<UNK>": UNK, **{word: i + 2 for i, word in enumerate(words)}}


def encode(text, vocab, max_length):
    ids = [vocab.get(token, UNK) for token in tokenize(text)[:max_length]]
    return ids + [PAD] * (max_length - len(ids))


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, channels=100, kernels=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, channels, k) for k in kernels])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(channels * len(kernels), 3)

    def forward(self, input_ids):
        x = self.embedding(input_ids).transpose(1, 2)
        x = torch.cat([torch.relu(conv(x)).amax(dim=2) for conv in self.convs], dim=1)
        return self.output(self.dropout(x))
