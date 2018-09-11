import collections
import os
import numpy as np
import argparse
import pdb
import tensorflow as tf


data_path = 'tutorial_data'
parser = argparse.ArgumentParser()
parser.add_argument('run_opt', type=int, default=1, help='An integer:1 to train, 2 to test')
parser.add_argument('--data_path', type=str, default=data_path, help='The path of the training data')
args = parser.parse_args()
if args.data_path:
    data_path = args.data_path

def read_words(filename):
    with tf.gfile.GFile(filename, 'r') as f:
        return f.read().replace('\n', '<eos>').split()

def build_vocab(filename):
    data = read_words(filename)
    # 对重复单词计数，并按出现次数从高到低排序
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

def file_to_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def load_data():
    train_path = os.path.join(data_path, 'ptb.train.txt')
    valid_path = os.path.join(data_path, 'ptb.valid.txt')
    test_path = os.path.join(data_path, 'ptb.test.txt')

    word_to_id = build_vocab(train_path)
    train_data = file_to_ids(train_path, word_to_id)
    valid_data = file_to_ids(valid_path, word_to_id)
    test_data = file_to_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reverse_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    # print(train_data[55:70])
    # print(word_to_id)
    # print(vocabulary)
    # print(' '.join(reverse_dictionary[x] for x in train_data[55:70]))
    return train_data, valid_data, test_data, vocabulary, reverse_dictionary

train_data, valid_data, test_data, vocabulary, reverse_dictionary = load_data()




