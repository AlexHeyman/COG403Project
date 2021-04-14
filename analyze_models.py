import os
import sys
import csv
import numpy as np
import torch
import torch.nn.functional as F
from train_model import SkipgramModel


num_superlative_words = 20
num_similar_embeddings = 20


def get_model_data(model_path, epoch_num):
    ev_file = open(os.path.join(model_path, 'embed_vocab.txt'), 'r', encoding='utf8')
    embed_vocab = ev_file.read().split()
    ev_file.close()
    embed_word2idx = {w: idx for (idx, w) in enumerate(embed_vocab)}

    pv_file = open(os.path.join(model_path, 'pred_vocab.txt'), 'r', encoding='utf8')
    pred_vocab = pv_file.read().split()
    pv_file.close()

    model = SkipgramModel(len(embed_vocab), len(pred_vocab))
    model.load_state_dict(torch.load(os.path.join(model_path, ('params_%s.pt' % epoch_num))))
    embeddings = model.linear1.weight.T
    return embed_vocab, embed_word2idx, embeddings


def record_most_similar(records, embeddings, i):
    if i in records:
        return
    n = embeddings.shape[0]
    similarities = [F.cosine_similarity(embeddings[i], embeddings[j], dim=0)\
                    for j in range(n)]
    records[i] = sorted(range(n),
                        key=lambda j: -similarities[j])[:num_similar_embeddings]


def analyze_models(model1_path, model2_path, epoch_num, norms_path):
    vocab1, word2idx1, embeddings1 = get_model_data(model1_path, epoch_num)
    vocab2, word2idx2, embeddings2 = get_model_data(model2_path, epoch_num)

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
    for word in vocab1:
        if word not in vocab2:
            continue
        untagged_word = word[:word.rindex('/')]
        if untagged_word in norms:
            norms_by_vocab_word[word] = norms[untagged_word]
    
    records1 = {}
    records2 = {}

    total_words_in_common = 0
    total_words_measured = 0

    def analyze_words_in_common(word):
        idx1 = word2idx1[word]
        idx2 = word2idx2[word]
        record_most_similar(records1, embeddings1, idx1)
        record_most_similar(records2, embeddings2, idx2)
        print(word)
        print('Model 1 | Model 2')
        words_in_common = 0
        words_measured = num_similar_embeddings
        words1 = []
        words2 = []
        for j in range(num_similar_embeddings):
            words1.append(vocab1[records1[idx1][j]])
            words2.append(vocab2[records2[idx2][j]])
            print(words1[j], words2[j])
        for j in range(num_similar_embeddings):
            if words1[j] in words2:
                words_in_common += 1
        print('Words in common: %d/%d' % (words_in_common, words_measured))
        print()
        return words_in_common, words_measured
    
    words_by_valence = sorted(norms_by_vocab_word.items(),
                              key=lambda x: x[1][0])
    print('Words with highest valence:')
    sup_words_in_common = 0
    sup_words_measured = 0
    for i in range(num_superlative_words):
        words_in_common, words_measured = analyze_words_in_common(words_by_valence[-(i + 1)][0])
        sup_words_in_common += words_in_common
        sup_words_measured += words_measured
        total_words_in_common += words_in_common
        total_words_measured += words_measured
    print('High-valence words in common: %d/%d' % (sup_words_in_common, sup_words_measured))
    print()
    print('Words with lowest valence:')
    sup_words_in_common = 0
    sup_words_measured = 0
    for i in range(num_superlative_words):
        words_in_common, words_measured = analyze_words_in_common(words_by_valence[i][0])
        sup_words_in_common += words_in_common
        sup_words_measured += words_measured
        total_words_in_common += words_in_common
        total_words_measured += words_measured
    print('Low-valence words in common: %d/%d' % (sup_words_in_common, sup_words_measured))

    print()
    
    words_by_arousal = sorted(norms_by_vocab_word.items(),
                              key=lambda x: x[1][1])
    print('Words with highest arousal:')
    sup_words_in_common = 0
    sup_words_measured = 0
    for i in range(num_superlative_words):
        words_in_common, words_measured = analyze_words_in_common(words_by_arousal[-(i + 1)][0])
        sup_words_in_common += words_in_common
        sup_words_measured += words_measured
        total_words_in_common += words_in_common
        total_words_measured += words_measured
    print('High-arousal words in common: %d/%d' % (sup_words_in_common, sup_words_measured))
    print()
    print('Words with lowest arousal:')
    sup_words_in_common = 0
    sup_words_measured = 0
    for i in range(num_superlative_words):
        words_in_common, words_measured = analyze_words_in_common(words_by_arousal[i][0])
        sup_words_in_common += words_in_common
        sup_words_measured += words_measured
        total_words_in_common += words_in_common
        total_words_measured += words_measured
    print('Low-arousal words in common: %d/%d' % (sup_words_in_common, sup_words_measured))
    print()
    
    words_by_dominance = sorted(norms_by_vocab_word.items(),
                              key=lambda x: x[1][2])
    print('Words with highest dominance:')
    sup_words_in_common = 0
    sup_words_measured = 0
    for i in range(num_superlative_words):
        words_in_common, words_measured = analyze_words_in_common(words_by_dominance[-(i + 1)][0])
        sup_words_in_common += words_in_common
        sup_words_measured += words_measured
        total_words_in_common += words_in_common
        total_words_measured += words_measured
    print('High-dominance words in common: %d/%d' % (sup_words_in_common, sup_words_measured))
    print()
    print('Words with lowest dominance:')
    sup_words_in_common = 0
    sup_words_measured = 0
    for i in range(num_superlative_words):
        words_in_common, words_measured = analyze_words_in_common(words_by_dominance[i][0])
        sup_words_in_common += words_in_common
        sup_words_measured += words_measured
        total_words_in_common += words_in_common
        total_words_measured += words_measured
    print('Low-dominance words in common: %d/%d' % (sup_words_in_common, sup_words_measured))
    print()
    print('Total words in common: %d/%d' % (total_words_in_common, total_words_measured))


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: python analyze_models.py [path to model 1 folder] [path to model 2 folder] [epoch number] [path to norms file]')
        exit()

    analyze_models(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
