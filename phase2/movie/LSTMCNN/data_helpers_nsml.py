# -*- coding: utf-8 -*-
import numpy as np
import re
import itertools
from collections import Counter
import gzip
import csv
import os

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
    """
    #string = re.sub(r"[^A-Za-z0-9\uAC00-\uD7A3(),!?\'\`]", " ", string)
    string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    #string = re.sub(r",", " , ", string)
    #######################################
    string = re.sub(r",", " ", string)
    string = re.sub(r"/", " ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    #string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

    x_text = all[1:, 2]
    print("origin msg: ", x_text[0])

    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]

    labels = np.array(all[1:, 0]).astype(np.int)
    print( "msg: ", x_text[0])
    print(len(x_text), len(labels))
    
    y = []
    for l in labels:
        value = [0] * 10
        value[l] = 1
        y.append(value)
    return [x_text, y]


def load_data_and_labels(data_path):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # data 불러오는 경로를 바꾸려면 이 부분을 수정한다
    data_review = os.path.join(data_path, 'train', 'train_data')
    data_label = os.path.join(data_path, 'train', 'train_label')

    # 영화리뷰 데이터를 읽고 preprocess까지 진행합니다
    with open(data_review, 'rt', encoding='utf-8') as f:
        reviews_list = f.readlines()
    # 영화리뷰 레이블을 읽고 preprocess까지 진행합니다.
    with open(data_label) as f:
        labels_list = [[np.float32(x) * 0.1] for x in f.readlines()]

    x_text = reviews_list
    print("origin msg: ", x_text[0])

    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]

    labels = labels_list
    print("msg: ", x_text[0])
    print(len(x_text), len(labels))

    # 이 부분이 KISTEP data에 맞게 labeling을 하는 부분이다
    # 0~31의 숫자를 onehot encoding으로 바꾸어준다
    y = labels

    return [x_text, y]


def load_data_and_labels_infer(raw_input):


    reviews_list = raw_input

    x_text = reviews_list
    print("origin msg: ", x_text[0])

    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]

    return x_text

def pad_sentences(sentences, padding_word='pad'):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    #sequence_length = max(len(x) for x in sentences)
    #print('sequence_length = {}'.format(sequence_length))
    sequence_length = 8
    padded_sentences = []
    length_list = []
    padding_word = 'pad'
    for sentence in sentences:
        #sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        if num_padding < 0:
            #print('you may increse sequence_length\t\t\tnow length = {}'.format(sequence_length))
            new_sentence = sentence[:sequence_length]
        else:
            new_sentence = sentence + [padding_word] * num_padding

        length_list.append(len(sentence))
        #num_padding = sequence_length - len(sentence)
        #new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)

    print('sentences length info :\tmean = {:.2f}\tmax = {}\tmin = {}\tstd = {:.2f}'.format(np.mean(length_list), max(length_list),
                                                                              min(length_list), np.std(length_list)))
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    vocabulary['unk'] = len(vocabulary)
    #vocabulary['pad'] = len(vocabulary)


    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)


    return [x, y]

def build_input_data_infer(sentences, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = []
    for sentence in sentences:
        xx = []
        for word in sentence:
            if word in vocabulary:
                xx.append(vocabulary[word])
            else:
                xx.append(vocabulary['unk'])
        x.append(xx)

    x = np.array(x)

    #x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])

    return x


def load_data(data_path):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(data_path)
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]


def load_data_infer(raw_input, vocabulary):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences = load_data_and_labels_infer(raw_input)
    sentences_padded = pad_sentences(sentences)
    #vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x = build_input_data_infer(sentences_padded, vocabulary)
    return x


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


class Vocab(object):
    def __init__(self, voc = None):

        self.voc = voc

