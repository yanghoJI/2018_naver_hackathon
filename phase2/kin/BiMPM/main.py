# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import sys
import time
import re
import tensorflow as tf
import json

from vocab_utils import Vocab
from SentenceMatchDataStream import SentenceMatchDataStream
from SentenceMatchModelGraph import SentenceMatchModelGraph
import namespace_utils

import pickle
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML


##for nsml##
def bind_model(sess, config, train_graph, word_vocab, char_vocab):
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
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        #preprocessed_data = preprocess(raw_data, config.strmaxlen)
        #raw_data = raw_data['data'][0]

        print('raw_data_ininferF = {}'.format(raw_data))

        #error interval
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


        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        #pred = sess.run(output_sigmoid, feed_dict={x: preprocessed_data})
        #clipped = np.array(pred > config.threshold, dtype=np.int)
        result = evaluation_kin(sess, valid_graph=train_graph, devDataStream=testStream)

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
        #if line.startswith('-'): continue
        items = re.split("\t", line)
        #label = items[0]
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
    parser.add_argument('--word_emb_dim', type=int, default=200, help='word_emb_dim')
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
    
    parser.add_argument('--config_path', default='./configs/quora.sample.config', type=str, help='Configuration file.')

    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default='0')

#     print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    args, unparsed = parser.parse_known_args()
    config = args
    sys.stdout.flush()
    print('training argument = {}'.format(config))
    #########################################################################main(FLAGS)

    # DONOTCHANGE: Reserved for nsml

    train_path = DATASET_PATH
    log_dir = config.model_dir


    char_vocab = None
    # if os.path.exists(best_path + ".index"):
    if config.mode == 'train':
        print('Collecting words, chars and labels ...')
        # (all_words, all_chars, all_labels, all_POSs, all_NERs) = collect_vocabs(train_path)
        (all_words, all_chars, all_labels, all_POSs, all_NERs) = collect_vocabs_kin(train_path)
        print('Number of words: {}'.format(len(all_words)))
        label_vocab = Vocab(fileformat='voc', voc=all_labels, dim=2)
        # label_vocab.dump_to_txt2(label_path)
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
    init_scale = 0.01

    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    global_step = tf.train.get_or_create_global_step()
    with tf.variable_scope("Model", reuse=None, initializer=initializer):
        train_graph = SentenceMatchModelGraph(num_classes, word_vocab=word_vocab, char_vocab=char_vocab,
                                              is_training=True, options=config, global_step=global_step)


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
    bind_model(sess=sess, config=config, train_graph=train_graph, word_vocab=word_vocab, char_vocab=char_vocab)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        print('Build SentenceMatchDataStream ... ')
        trainDataStream = SentenceMatchDataStream(train_path, word_vocab=word_vocab, char_vocab=char_vocab,
                                                  label_vocab=label_vocab,
                                                  isShuffle=True, isLoop=True, isSort=True, options=config)
        print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
        print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))

        # training
        print("start train!!")
        ######################## train(sess, saver=0, train_graph=train_graph, valid_graph=0, trainDataStream=trainDataStream, devDataStream=0, options=FLAGS, best_path=0, word_vocab=word_vocab, char_vocab=char_vocab)

        best_accuracy = -1
        best_epoch = -1
        # options.max_epochs = 140


        for epoch in range(config.max_epochs):
            print('Train in epoch %d' % epoch)
            # training
            trainDataStream.shuffle()
            num_batch = trainDataStream.get_num_batch()
            start_time = time.time()
            total_loss = 0
            for batch_index in range(num_batch):  # for each batch
                cur_batch = trainDataStream.get_batch(batch_index)
                feed_dict = train_graph.create_feed_dict(cur_batch, is_training=True)
                _, loss_value = sess.run([train_graph.train_op, train_graph.loss], feed_dict=feed_dict)
                total_loss += loss_value

                if batch_index % 100 == 0:
                    print('{} '.format(batch_index), end="")
                    sys.stdout.flush()

            print()
            duration = time.time() - start_time
            print('Epoch %d: loss = %.4f (%.3f sec)' % (epoch, total_loss / num_batch, duration))
            nsml.save(epoch)
        #dummy_raw = ['국채 발행 이유?	국채를 많이 발행하면 왜 수요가 떨어져요?', '국채 발행 이유?	나라가 국채를 발행할때']
        #result = nsml.infer(dummy_raw)
        #print(result)

        # print('infer cheak')
        # data_path = os.path.join(train_path, 'train', 'train_data')
        # data_infile = open(data_path, 'rt', encoding='utf-8')
        # datas = data_infile.readlines()


    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res
        print(res)

