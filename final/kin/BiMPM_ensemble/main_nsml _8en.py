# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import sys
import time
import re
import tensorflow as tf
import json
import numpy as np

from vocab_utils import Vocab
from SentenceMatchDataStream import SentenceMatchDataStream
from SentenceMatchModelGraph import SentenceMatchModelGraph
import namespace_utils

import pickle
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML


##for nsml##
def bind_model(sess, config, modellist, word_vocab, char_vocab):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))
        with open(os.path.join(dir_name, 'voc.pkl'), 'wb') as f:
            pickle.dump(word_vocab, f)
            pickle.dump(label_vocab, f)
            pickle.dump(char_vocab, f)


    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

        with open(os.path.join(dir_name, 'voc.pkl'), 'rb') as f:
            word_vocabf = pickle.load(f)
            label_vocabf = pickle.load(f)
            char_vocabf = pickle.load(f)
        word_vocab.append(word_vocabf)
        char_vocab.append(char_vocabf)
        label_vocab.append(label_vocabf)


    def infer(raw_data, **kwargs):
        """
        sess, config, raw_data, train_graph, word_vocab, char_vocab
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """

        if type(word_vocab) is list:
            testStream = SentenceMatchDataStream(raw_data, word_vocab=word_vocab[0], char_vocab=char_vocab[0],
                                                     label_vocab=None,
                                                     isShuffle=False, isLoop=False, isSort=False, options=config,
                                                     raw_data=True)
        else:
            testStream = SentenceMatchDataStream(raw_data, word_vocab=word_vocab, char_vocab=char_vocab,
                                                 label_vocab=None,
                                                 isShuffle=False, isLoop=False, isSort=False, options=config,
                                                 raw_data=True)

        result = evaluation_kin_en(sess, model_list=modellist, devDataStream=testStream)

        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        #return list(zip(pred.flatten(), clipped.flatten()))
        return result

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)




def collect_vocabs_kin(train_path, with_POS=False, with_NER=False):
    data_path = os.path.join(train_path, 'train', 'train_data')
    label_path = os.path.join(train_path, 'train', 'train_label')
    all_labels = set()
    all_words = set()
    all_POSs = None
    all_NERs = None
    if with_POS: all_POSs = set()
    if with_NER: all_NERs = set()

    data_infile = open(data_path, 'rt', encoding = 'utf-8')
    label_infile = open(label_path, 'rt', encoding = 'utf-8')
    for line in data_infile:
        line = line.strip()
        items = re.split("\t", line)
        sentence1 = re.split("\\s+",items[0].lower())
        sentence2 = re.split("\\s+",items[1].lower())
        #all_labels.add(label)
        all_words.update(sentence1)
        all_words.update(sentence2)
        if with_POS:
            all_POSs.update(re.split("\\s+",items[3]))
            all_POSs.update(re.split("\\s+",items[4]))
        if with_NER:
            all_NERs.update(re.split("\\s+",items[5]))
            all_NERs.update(re.split("\\s+",items[6]))
    data_infile.close()

    for line in label_infile:
        line = line.strip()
        all_labels.add(line)
    label_infile.close()

    all_chars = set()
    for word in all_words:
        for char in word:
            all_chars.add(char)
    return (all_words, all_chars, all_labels, all_POSs, all_NERs)

def output_probs(probs, label_vocab):
    out_string = ""
    for i in range(probs.size):
        out_string += " {}:{}".format(label_vocab.getWord(i), probs[i])
    return out_string.strip()

def evaluation(sess, valid_graph, devDataStream, outpath=None, label_vocab=None):
    if outpath is not None:
        result_json = {}
    total = 0
    correct = 0
    for batch_index in range(devDataStream.get_num_batch()):  # for each batch
        cur_batch = devDataStream.get_batch(batch_index)
        total += cur_batch.batch_size
        feed_dict = valid_graph.create_feed_dict(cur_batch, is_training=True)
        #print("label = {}".format(feed_dict[real]))
        [cur_correct, probs, predictions] = sess.run([valid_graph.eval_correct, valid_graph.prob, valid_graph.predictions], feed_dict=feed_dict)
        correct += cur_correct
        #print("cur_correct={}, probs={}, predictions={}".format(cur_correct, probs, predictions))
        if outpath is not None:
            for i in range(cur_batch.batch_size):
                (label, sentence1, sentence2, _, _, _, _, _, cur_ID) = cur_batch.instances[i]
                result_json[cur_ID] = {
                    "ID": cur_ID,
                    "truth": label,
                    "sent1": sentence1,
                    "sent2": sentence2,
                    "prediction": label_vocab.getWord(predictions[i]),
                    "probs": output_probs(probs[i], label_vocab),
                }
    accuracy = correct / float(total) * 100
    if outpath is not None:
        with open(outpath, 'w') as outfile:
            json.dump(result_json, outfile)
    return accuracy

def evaluation_kin(sess, valid_graph, devDataStream, outpath=None, label_vocab=None):
    if outpath is not None:
        result_json = {}
    total = 0
    correct = 0
    all_probs = []
    all_predictions = []
    for batch_index in range(devDataStream.get_num_batch()):  # for each batch
        cur_batch = devDataStream.get_batch(batch_index)
        total += cur_batch.batch_size
        feed_dict = valid_graph.create_feed_dict(cur_batch, is_training=True)
        #print("label = {}".format(feed_dict[real]))
        [cur_correct, probs, predictions] = sess.run([valid_graph.eval_correct, valid_graph.prob, valid_graph.predictions], feed_dict=feed_dict)
        probs = probs[:,1]
        all_probs.extend(list(probs))
        all_predictions.extend(list(predictions))
        correct += cur_correct
    result = list(zip(all_probs, all_predictions))

    return result

def evaluation_kin_en(sess, model_list, devDataStream, outpath=None, label_vocab=None):
    if outpath is not None:
        result_json = {}
    total = 0

    prob1_total = []
    prob2_total = []
    prob3_total = []
    prob4_total = []
    prob5_total = []
    prob6_total = []
    prob7_total = []
    prob8_total = []


    for batch_index in range(devDataStream.get_num_batch()):  # for each batch
        cur_batch = devDataStream.get_batch(batch_index)
        total += cur_batch.batch_size
        feed_dict_1 = model_list[0].create_feed_dict(cur_batch, is_training=True)
        feed_dict_2 = model_list[1].create_feed_dict(cur_batch, is_training=True)
        feed_dict_3 = model_list[2].create_feed_dict(cur_batch, is_training=True)
        feed_dict_4 = model_list[3].create_feed_dict(cur_batch, is_training=True)
        feed_dict_5 = model_list[4].create_feed_dict(cur_batch, is_training=True)
        feed_dict_6 = model_list[5].create_feed_dict(cur_batch, is_training=True)
        feed_dict_7 = model_list[6].create_feed_dict(cur_batch, is_training=True)
        feed_dict_8 = model_list[7].create_feed_dict(cur_batch, is_training=True)


        prob1 = sess.run(model_list[0].prob, feed_dict=feed_dict_1)
        prob1_total.extend(list(prob1[:, 1]))

        prob2 = sess.run(model_list[1].prob, feed_dict=feed_dict_2)
        prob2_total.extend(list(prob2[:, 1]))

        prob3 = sess.run(model_list[2].prob, feed_dict=feed_dict_3)
        prob3_total.extend(list(prob3[:, 1]))

        prob4 = sess.run(model_list[3].prob, feed_dict=feed_dict_4)
        prob4_total.extend(list(prob4[:, 1]))

        prob5 = sess.run(model_list[4].prob, feed_dict=feed_dict_5)
        prob5_total.extend(list(prob5[:, 1]))

        prob6 = sess.run(model_list[5].prob, feed_dict=feed_dict_6)
        prob6_total.extend(list(prob6[:, 1]))

        prob7 = sess.run(model_list[6].prob, feed_dict=feed_dict_7)
        prob7_total.extend(list(prob7[:, 1]))

        prob8 = sess.run(model_list[7].prob, feed_dict=feed_dict_8)
        prob8_total.extend(list(prob8[:, 1]))




    prob_list = [prob1_total, prob2_total, prob3_total, prob4_total, prob5_total, prob6_total, prob7_total, prob8_total]

    ## en prediction
    probsum = np.expand_dims(np.array(prob_list[0]), 1)

    if len(prob_list) > 1:
        for idx in range(1, len(prob_list)):
            probsum = np.concatenate((probsum, np.expand_dims(np.array(prob_list[idx]), 1)), axis=1)

    avg_probs = np.mean(probsum, axis=1)
    pridictions = avg_probs > 0.5
    pridictions = pridictions.astype(int)

    result = list(zip(list(avg_probs), list(pridictions)))

    return result

def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]

def enrich_options(options):
    #if not options.__dict__.has_key("in_format"):
    if not "in_format" in options.__dict__:
        options.__dict__["in_format"] = 'tsv'

    return options

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default="./dataset/",help='Path to the train set.')
    parser.add_argument('--dev_path', type=str, default="./dataset/",help='Path to the dev set.')
    parser.add_argument('--test_path', type=str, default="./dataset/",help='Path to the test set.')
    parser.add_argument('--word_vec_path', type=str, default="./wordvec11.txt",help='Path the to pre-trained word vector model.')
    parser.add_argument('--model_dir', type=str, default="./logs",help='Directory to save model files.')
    parser.add_argument('--batch_size', type=int, default=80, help='Number of instances in each batch.')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--lambda_l2', type=float, default=0.0, help='The coefficient of L2 regularizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.35, help='Dropout ratio.')
    parser.add_argument('--max_epochs', type=int, default=40, help='Maximum epochs for training.')
    parser.add_argument('--optimize_type', type=str, default='adam', help='Optimizer type.')

    parser.add_argument('--char_emb_dim', type=int, default=20, help='Number of dimension for character embeddings.')
    parser.add_argument('--word_emb_dim', type=int, default=150, help='word_emb_dim')
    parser.add_argument('--char_lstm_dim', type=int, default=40, help='Number of dimension for character-composed embeddings.')
    parser.add_argument('--context_lstm_dim', type=int, default=100, help='Number of dimension for context representation layer.')
    parser.add_argument('--aggregation_lstm_dim', type=int, default=100, help='Number of dimension for aggregation layer.')
    parser.add_argument('--max_char_per_word', type=int, default=10, help='Maximum number of characters for each word.')
    parser.add_argument('--max_sent_length', type=int, default=30, help='Maximum number of words within each sentence.')
    parser.add_argument('--aggregation_layer_num', type=int, default=1, help='Number of LSTM layers for aggregation layer.')
    parser.add_argument('--context_layer_num', type=int, default=1, help='Number of LSTM layers for context representation layer.')
    parser.add_argument('--highway_layer_num', type=int, default=1, help='Number of highway layers.')
    parser.add_argument('--suffix', type=str, default='quora', help='Suffix of the model name.')
    parser.add_argument('--fix_word_vec', default=False, help='Fix pre-trained word embeddings during training.', action='store_true')
    parser.add_argument('--with_highway', default=True, help='Utilize highway layers.', action='store_true')
    parser.add_argument('--with_match_highway', default=True, help='Utilize highway layers for matching layer.', action='store_true')
    parser.add_argument('--with_aggregation_highway', default=True, help='Utilize highway layers for aggregation layer.', action='store_true')
    parser.add_argument('--with_full_match', default=True, help='With full matching.', action='store_true')
    parser.add_argument('--with_maxpool_match', default=False, help='With maxpooling matching', action='store_true')
    parser.add_argument('--with_attentive_match', default=True, help='With attentive matching', action='store_true')
    parser.add_argument('--with_max_attentive_match', default=False, help='With max attentive matching.', action='store_true')
    parser.add_argument('--isLower', default=True, help='isLower', action='store_true')

    parser.add_argument('--att_type', type=str, default="symmetric", help='att_type')
    parser.add_argument('--att_dim', type=int, default=50, help='att_dim')
    parser.add_argument('--with_cosine', default=True, help='with_cosine', action='store_true')
    parser.add_argument('--with_mp_cosine', default=True, help='with_mp_cosine', action='store_true')
    parser.add_argument('--cosine_MP_dim', type=int, default=5, help='cosine_MP_dim')
    parser.add_argument('--with_char', default=True, help='with_char', action='store_true')
    parser.add_argument('--grad_clipper', type=float, default=10.0, help='grad_clipper')
    parser.add_argument('--use_cudnn', default=True, help='use_cudnn', action='store_true')
    parser.add_argument('--with_moving_average', default=False, help='with_moving_average', action='store_true')
    parser.add_argument('--bagging', type=float, default=0.93, help='bagging')

    
    parser.add_argument('--config_path', default='./configs/quora.sample.config', type=str, help='Configuration file.')

    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default='0')

    args, unparsed = parser.parse_known_args()
    config = args
    sys.stdout.flush()
    print('training argument = {}'.format(config))
    #########################################################################main(FLAGS)

    # DONOTCHANGE: Reserved for nsml

    train_path = DATASET_PATH
    #train_path = './dataset/'
    log_dir = config.model_dir


    char_vocab = None
    # if os.path.exists(best_path + ".index"):
    if config.mode == 'train':
        print('Collecting words, chars and labels ...')
        (all_words, all_chars, all_labels, all_POSs, all_NERs) = collect_vocabs_kin(train_path)
        print('Number of words: {}'.format(len(all_words)))
        label_vocab = Vocab(fileformat='voc', voc=all_labels, dim=2)
        if label_vocab.id2word[0] != '0':
            print('{}'.format('#'*15))
            print('arange label voc !!')
            print('{}'.format('#' * 15))
            label_vocab.id2word[0] = '0'
            label_vocab.id2word[1] = '1'
            label_vocab.word2id['0'] = 0
            label_vocab.word2id['1'] = 1

        word_vocab = Vocab(fileformat='voc', voc=all_words, dim=config.word_emb_dim)
        

        if config.with_char:
            print('Number of chars: {}'.format(len(all_chars)))
            char_vocab = Vocab(fileformat='voc', voc=all_chars, dim=config.char_emb_dim)
            # char_vocab.dump_to_txt2(char_path)
    else:
        print('test seq ')
        word_vocab = []
        label_vocab = []
        char_vocab = []

    num_classes = 2
    init_scale = 0.1

    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    global_step = tf.train.get_or_create_global_step()

    with tf.variable_scope("Model_1", reuse=None, initializer=initializer):
        train_graph_1 = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                              is_training=True, options=config, global_step=global_step)

    with tf.variable_scope("Model_2", reuse=None, initializer=initializer):
        train_graph_2 = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                is_training=True, options=config, global_step=global_step)

    with tf.variable_scope("Model_3", reuse=None, initializer=initializer):
        train_graph_3 = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                is_training=True, options=config, global_step=global_step)

    with tf.variable_scope("Model_4", reuse=None, initializer=initializer):
        train_graph_4 = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                is_training=True, options=config, global_step=global_step)

    with tf.variable_scope("Model_5", reuse=None, initializer=initializer):
        train_graph_5 = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                is_training=True, options=config, global_step=global_step)

    with tf.variable_scope("Model_6", reuse=None, initializer=initializer):
        train_graph_6 = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                is_training=True, options=config, global_step=global_step)

    with tf.variable_scope("Model_7", reuse=None, initializer=initializer):
        train_graph_7 = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                is_training=True, options=config, global_step=global_step)

    with tf.variable_scope("Model_8", reuse=None, initializer=initializer):
        train_graph_8 = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                                is_training=True, options=config, global_step=global_step)



    model_list = [train_graph_1, train_graph_2, train_graph_3, train_graph_4, train_graph_5, train_graph_6, train_graph_7, train_graph_8]


    print('start initialize')
    initializer = tf.global_variables_initializer()
    print('start sess = tf.Session()')
    t1 = time.time()
    #sess = tf.Session()
    sess = tf.InteractiveSession()
    print('sess.run(initializer)')
    sess.run(initializer)
    t2 = time.time()
    print('sess open time = {}'.format(t2 - t1))

    print('bind_model')
    bind_model(sess=sess, config=config, modellist=model_list, word_vocab=word_vocab, char_vocab=char_vocab)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        print('Build SentenceMatchDataStream ... ')
        trainDataStream1 = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                   label_vocab=label_vocab,
                                                   isShuffle=True, isLoop=True, isSort=True, options=config,
                                                   bagging=config.bagging)

        trainDataStream2 = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                   label_vocab=label_vocab,
                                                   isShuffle=True, isLoop=True, isSort=True, options=config,
                                                   bagging=config.bagging)

        trainDataStream3 = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                   label_vocab=label_vocab,
                                                   isShuffle=True, isLoop=True, isSort=True, options=config,
                                                   bagging=config.bagging)

        trainDataStream4 = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                   label_vocab=label_vocab,
                                                   isShuffle=True, isLoop=True, isSort=True, options=config,
                                                   bagging=config.bagging)

        trainDataStream5 = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                   label_vocab=label_vocab,
                                                   isShuffle=True, isLoop=True, isSort=True, options=config,
                                                   bagging=config.bagging)

        trainDataStream6 = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                   label_vocab=label_vocab,
                                                   isShuffle=True, isLoop=True, isSort=True, options=config,
                                                   bagging=config.bagging)

        trainDataStream7 = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                   label_vocab=label_vocab,
                                                   isShuffle=True, isLoop=True, isSort=True, options=config,
                                                   bagging=config.bagging)

        trainDataStream8 = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                   label_vocab=label_vocab,
                                                   isShuffle=True, isLoop=True, isSort=True, options=config,
                                                   bagging=config.bagging)




        print('Number of instances in trainDataStream: {}'.format(trainDataStream1.get_num_instance()))
        print('Number of batches in trainDataStream: {}'.format(trainDataStream1.get_num_batch()))

        # training
        print("start train!!")
        best_accuracy = -1
        best_epoch = -1
        # options.max_epochs = 140


        for epoch in range(config.max_epochs):
            print('Train in epoch %d' % epoch)
            nsml.save(epoch)
            # training
            trainDataStream1.shuffle()
            trainDataStream2.shuffle()
            trainDataStream3.shuffle()
            trainDataStream4.shuffle()
            trainDataStream5.shuffle()
            trainDataStream6.shuffle()
            trainDataStream7.shuffle()
            trainDataStream8.shuffle()



            num_batch = trainDataStream1.get_num_batch()
            start_time = time.time()
            total_loss_1 = 0
            total_loss_2 = 0
            total_loss_3 = 0
            total_loss_4 = 0
            total_loss_5 = 0
            total_loss_6 = 0
            total_loss_7 = 0
            total_loss_8 = 0


            for batch_index in range(num_batch):  # for each batch
                cur_batch1 = trainDataStream1.get_batch(batch_index)
                cur_batch2 = trainDataStream2.get_batch(batch_index)
                cur_batch3 = trainDataStream3.get_batch(batch_index)
                cur_batch4 = trainDataStream4.get_batch(batch_index)
                cur_batch5 = trainDataStream5.get_batch(batch_index)
                cur_batch6 = trainDataStream6.get_batch(batch_index)
                cur_batch7 = trainDataStream7.get_batch(batch_index)
                cur_batch8 = trainDataStream8.get_batch(batch_index)


                feed_dict_1 = train_graph_1.create_feed_dict(cur_batch1, is_training=True)
                feed_dict_2 = train_graph_2.create_feed_dict(cur_batch2, is_training=True)
                feed_dict_3 = train_graph_3.create_feed_dict(cur_batch3, is_training=True)
                feed_dict_4 = train_graph_4.create_feed_dict(cur_batch4, is_training=True)
                feed_dict_5 = train_graph_5.create_feed_dict(cur_batch5, is_training=True)
                feed_dict_6 = train_graph_6.create_feed_dict(cur_batch6, is_training=True)
                feed_dict_7 = train_graph_7.create_feed_dict(cur_batch7, is_training=True)
                feed_dict_8 = train_graph_8.create_feed_dict(cur_batch8, is_training=True)



                _1, loss_value_1 = sess.run([train_graph_1.train_op, train_graph_1.loss], feed_dict=feed_dict_1)
                _2, loss_value_2 = sess.run([train_graph_2.train_op, train_graph_2.loss], feed_dict=feed_dict_2)
                _3, loss_value_3 = sess.run([train_graph_3.train_op, train_graph_3.loss], feed_dict=feed_dict_3)
                _4, loss_value_4 = sess.run([train_graph_4.train_op, train_graph_4.loss], feed_dict=feed_dict_4)
                _5, loss_value_5 = sess.run([train_graph_5.train_op, train_graph_5.loss], feed_dict=feed_dict_5)
                _6, loss_value_6 = sess.run([train_graph_6.train_op, train_graph_6.loss], feed_dict=feed_dict_6)
                _7, loss_value_7 = sess.run([train_graph_7.train_op, train_graph_7.loss], feed_dict=feed_dict_7)
                _8, loss_value_8 = sess.run([train_graph_8.train_op, train_graph_8.loss], feed_dict=feed_dict_8)



                total_loss_1 += loss_value_1
                total_loss_2 += loss_value_2
                total_loss_3 += loss_value_3
                total_loss_4 += loss_value_4
                total_loss_5 += loss_value_5
                total_loss_6 += loss_value_6
                total_loss_7 += loss_value_7
                total_loss_8 += loss_value_8



                if batch_index % 100 == 0:
                    print('{} '.format(batch_index), end="")
                    sys.stdout.flush()


            print()
            duration = time.time() - start_time
            total_loss = (total_loss_1 + total_loss_2 + total_loss_3+ total_loss_4 + total_loss_5+ total_loss_6 + total_loss_7+ total_loss_8)/3
            print('Epoch %d: avg_loss = %.4f (%.3f sec)' % (epoch, total_loss / num_batch, duration))
            #nsml.save(epoch)



    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)

