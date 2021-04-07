import sys
from collections import OrderedDict
import csv
import torch
import torch.nn.functional as F
from train_model import SkipgramModel


num_superlative_words = 10
num_similar_embeddings = 10


def record_most_similar(records, embeddings, i):
    if i in records:
        return
    n = embeddings.shape[0]
    similarities = [F.cosine_similarity(embeddings[i], embeddings[j], dim=0)\
                    for j in range(n)]
    records[i] = sorted(range(n),
                        key=lambda j: -similarities[j])[:num_similar_embeddings]


def analyze_model(vocab_path, params_path, norms_path):
    vocab_file = open(vocab_path, 'r', encoding='utf8')
    vocabulary = vocab_file.read().split(' ')
    vocab_file.close()
    vocabulary_size = len(vocabulary)
    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}

    model = SkipgramModel(vocabulary_size)
    model.load_state_dict(torch.load(params_path))
    embeddings = model.linear1.weight.T

    norms = {}
    with open(norms_path, newline='') as norms_file:
        reader = csv.reader(norms_file, delimiter=' ', quotechar='|')
        for row in reader:
            row = row[0].split(',')
            if len(row) < 9 or len(row[0]) == 0:
                continue
            # Mean sums for valence, arousal, and dominance, respectively
            norms[row[1]] = [float(row[2]), float(row[5]), float(row[8])]

    norms_by_vocab_word = {}
    for i in range(vocabulary_size):
        word = vocabulary[i]
        untagged_word = word[:word.rindex('/')]
        if untagged_word in norms:
            norms_by_vocab_word[word] = norms[untagged_word]

    most_similar_records = {}
    
    words_by_valence = sorted(norms_by_vocab_word.items(),
                              key=lambda x: x[1][0])
    print('Words with highest valence:')
    for i in range(num_superlative_words):
        word = words_by_valence[-(i + 1)][0]
        idx = word2idx[word]
        record_most_similar(most_similar_records, embeddings, idx)
        print(word)
        print(list(vocabulary[j] for j in most_similar_records[idx]))
    print()
    print('Words with lowest valence:')
    for i in range(num_superlative_words):
        word = words_by_valence[i][0]
        idx = word2idx[word]
        record_most_similar(most_similar_records, embeddings, idx)
        print(word)
        print([vocabulary[j] for j in most_similar_records[idx]])
    print()
    
    words_by_arousal = sorted(norms_by_vocab_word.items(),
                              key=lambda x: x[1][1])
    print('Words with highest arousal:')
    for i in range(num_superlative_words):
        word = words_by_arousal[-(i + 1)][0]
        idx = word2idx[word]
        record_most_similar(most_similar_records, embeddings, idx)
        print(word)
        print([vocabulary[j] for j in most_similar_records[idx]])
    print()
    print('Words with lowest arousal:')
    for i in range(num_superlative_words):
        word = words_by_arousal[i][0]
        idx = word2idx[word]
        record_most_similar(most_similar_records, embeddings, idx)
        print(word)
        print([vocabulary[j] for j in most_similar_records[idx]])
    print()
    
    words_by_dominance = sorted(norms_by_vocab_word.items(),
                              key=lambda x: x[1][2])
    print('Words with highest dominance:')
    for i in range(num_superlative_words):
        word = words_by_dominance[-(i + 1)][0]
        idx = word2idx[word]
        record_most_similar(most_similar_records, embeddings, idx)
        print(word)
        print([vocabulary[j] for j in most_similar_records[idx]])
    print()
    print('Words with lowest dominance:')
    for i in range(num_superlative_words):
        word = words_by_dominance[i][0]
        idx = word2idx[word]
        record_most_similar(most_similar_records, embeddings, idx)
        print(word)
        print([vocabulary[j] for j in most_similar_records[idx]])
    print()


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python analyze_model.py [path to model vocabulary] [path to model parameters] [path to Warriner norms csv file]')
        exit()

    analyze_model(sys.argv[1], sys.argv[2], sys.argv[3])
