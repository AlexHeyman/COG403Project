import sys
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


embedding_dim = 300
min_word_frequency = 10
num_epochs = 5
learning_rate = 0.001
window_size = 2
batch_size = 1000


class SkipgramModel(nn.Module):
    
    def __init__(self, embed_vocab_size, pred_vocab_size):
        super(SkipgramModel, self).__init__()
        self.embed_vocab_size = embed_vocab_size
        self.pred_vocab_size = pred_vocab_size
        self.linear1 = nn.Linear(embed_vocab_size, embedding_dim, bias=False)
        self.linear2 = nn.Linear(embedding_dim, pred_vocab_size, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.one_hot(x, num_classes=self.embed_vocab_size).float()
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.log_softmax(x)
        return x


def get_input(input_path):
    embed_vocab = OrderedDict()
    pred_vocab = OrderedDict()
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
            for i in range(len(words)):
                if words[i].endswith('/PROPN'):
                    words[i] = '/PROPN'
                elif not words[i].endswith('/PUNCT'):
                    if words[i] in embed_vocab:
                        embed_vocab[words[i]] += 1
                    else:
                        embed_vocab[words[i]] = 1
                
                if words[i] in pred_vocab:
                    pred_vocab[words[i]] += 1
                else:
                    pred_vocab[words[i]] = 1
            
            corpus[filename] = words

    for word in list(embed_vocab):
        if embed_vocab[word] < min_word_frequency:
            del embed_vocab[word]
    
    return embed_vocab, pred_vocab, corpus


def train_model(input_path, output_path):
    print('Retrieving input')
    embed_vocab, pred_vocab, corpus = get_input(input_path)
    
    ev_output_file = open(os.path.join(output_path, 'embed_vocab.txt'), 'w', encoding='utf8')
    ev_output_file.write(' '.join(embed_vocab))
    ev_output_file.close()

    pv_output_file = open(os.path.join(output_path, 'pred_vocab.txt'), 'w', encoding='utf8')
    pv_output_file.write(' '.join(pred_vocab))
    pv_output_file.close()

    print('Corpus size:', sum(len(story) for story in corpus.values()))
    
    embed_word2idx = {w: idx for (idx, w) in enumerate(embed_vocab)}
    embed_vocab_size = len(embed_vocab)
    print('Embedding vocab size:', embed_vocab_size)

    pred_word2idx = {w: idx for (idx, w) in enumerate(pred_vocab)}
    pred_vocab_size = len(pred_vocab)
    print('Prediction vocab size:', pred_vocab_size)

    print('Getting contexts')
    center_idxs = []
    context_idxs = []
    for _, story in corpus.items():
        pred_indices = [pred_word2idx[word] for word in story]
        for center_word_pos in range(len(story)):
            if story[center_word_pos] in embed_vocab:
                embed_index = embed_word2idx[story[center_word_pos]]
                for w in range(-window_size, window_size + 1):
                    context_word_pos = center_word_pos + w
                    if context_word_pos < 0 or context_word_pos >= len(story)\
                       or center_word_pos == context_word_pos:
                        continue
                    center_idxs.append(embed_index)
                    context_idxs.append(pred_indices[context_word_pos])
    num_pairs = len(center_idxs)
    print('Number of context pairs:', num_pairs)
    
    print('Training model')
    model = SkipgramModel(embed_vocab_size, pred_vocab_size)
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
    
        torch.save(model.state_dict(),
                   os.path.join(output_path, ('params_%d.pt' % (epoch + 1))))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python train_model.py [path to input folder] [path to output folder]')
        exit()

    train_model(sys.argv[1], sys.argv[2])
