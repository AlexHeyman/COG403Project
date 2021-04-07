import sys
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


embedding_dim = 300
num_epochs = 5
learning_rate = 0.01
window_size = 2
batch_size = 1000


class SkipgramModel(nn.Module):
    
    def __init__(self, vocabulary_size):
        super(SkipgramModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.linear1 = nn.Linear(vocabulary_size, embedding_dim, bias=False)
        self.linear2 = nn.Linear(embedding_dim, vocabulary_size, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.one_hot(x, num_classes=self.vocabulary_size).float()
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.log_softmax(x)
        return x


def get_input(input_path):
    vocabulary = OrderedDict()
    corpus = OrderedDict()
    for root, dirs, filenames in os.walk(input_path):
        for i in range(len(filenames)):
            filename = filenames[i]
            if not filename.endswith('.txt'):
                continue
            file = open(os.path.join(root, filename), 'r', encoding='utf8')
            data = file.read()
            file.close()
            words = data.split()
            corpus[filename] = words
            for word in words:
                if word in vocabulary:
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1
    
    return vocabulary, corpus


def train_model(input_path, vocab_output_path, params_output_path):
    print('Retrieving input')
    vocabulary, corpus = get_input(input_path)
    
    vocab_output_file = open(vocab_output_path, 'w', encoding='utf8')
    vocab_output_file.write(' '.join(vocabulary))
    vocab_output_file.close()
    
    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
    vocabulary_size = len(vocabulary)
    print('Vocabulary size:', vocabulary_size)
    
    print('Getting contexts')
    center_idxs = []
    context_idxs = []
    for _, story in corpus.items():
        indices = [word2idx[word] for word in story]
        for center_word_pos in range(len(indices)):
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                if context_word_pos < 0 or context_word_pos >= len(indices)\
                   or center_word_pos == context_word_pos:
                    continue
                center_idxs.append(indices[center_word_pos])
                context_idxs.append(indices[context_word_pos])
    num_pairs = len(center_idxs)
    print('Number of context pairs:', num_pairs)
    
    print('Training model')
    model = SkipgramModel(vocabulary_size)
    trainset = torch.utils.data.TensorDataset(torch.tensor(center_idxs).long(),
                                              torch.tensor(context_idxs).long())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        print('Epoch %d of %d' % (epoch + 1, num_epochs))
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 0 and i != 0:
                print('[%d, %d] loss: %.3f' %
                      (epoch + 1, i * batch_size, running_loss / 100))
                running_loss = 0.0
    
    torch.save(model.state_dict(), params_output_path)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python train_model.py [path to input folder] [path to output vocab] [path to output trained model parameters]')
        exit()

    train_model(sys.argv[1], sys.argv[2], sys.argv[3])
