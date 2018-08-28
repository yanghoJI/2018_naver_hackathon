# -*- coding: utf-8 -*-


import numpy as np
import os
import time
import datetime
import data_helpers_nsml as data_helpers
from text_cnn_nsml import TextCNN
import argparse
import tensorflow as tf

import pickle
import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML
#from dataset import KinQueryDataset, preprocess


def bind_model(sess, config, model, vocabulary):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))
        with open(os.path.join(dir_name, 'voc.pkl'), 'wb') as f:
            pickle.dump(vocabulary, f)


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
            vocabularyf = pickle.load(f)

        vocabulary.append(vocabularyf)


    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """

        preprocessed_data = data_helpers.load_data_infer(raw_data, vocabulary[0])

        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        feed_dict = {
            model.input_x: preprocessed_data,
            model.dropout_keep_prob: 1.0
        }
        pred = sess.run(model.scores, feed_dict=feed_dict)
        pred = pred * 10
        pred = pred.flatten()
        con = np.zeros(len(pred))
        result = list(zip(con, pred))

        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        # [(1, 7.4), (1, 2.3)]
        return result

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)




if __name__ == '__main__':

    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: config.dropout_keep_prob
        }

        _, step,  loss = sess.run(
            [train_op, global_step,  cnn.loss],
            feed_dict)
        accuracy = 0
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        return loss


    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            cnn.input_x: x_batch,
            cnn.input_y: y_batch,
            cnn.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy, true_val, results = sess.run(
            [global_step, dev_summary_op, cnn.loss, cnn.accuracy, dev_true_val, dev_results],
            feed_dict)

        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        print(true_val)  # 정답과 분류 결과를 비교하기 위해 디버깅 출력하는 부분이다
        print(results)



    # Parameters
    # ==================================================

    parser = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default='0')

    # Model Hyperparameters
    parser.add_argument('--filter_sizes', type=str, default="3,4,5",help="Comma-separated filter sizes (default: '3,4,5')")
    parser.add_argument('--embedding_dim', type=int, default=100, help="Dimensionality of character embedding (default: 128)")
    parser.add_argument('--num_filters', type=int, default=128, help="Number of filters per filter size (default: 128)")
    parser.add_argument('--dropout_keep_prob', type=float, default=0.8, help="Dropout keep probability (default: 0.5)")
    parser.add_argument('--l2_reg_lambda', type=float, default=0.001, help="L2 regularizaion lambda (default: 0.0)")
    parser.add_argument('--sequence_length', type=int, default=18, help="sequence_length (default: 18)")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch Size (default: 64)")
    parser.add_argument('--num_epochs', type=int, default=3, help="Number of training epochs (default: 200)")
    parser.add_argument('--evaluate_every', type=int, default=500, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument('--checkpoint_every', type=int, default=20, help="Save model after this many steps (default: 100)")


    # Misc Parameters
    parser.add_argument('--allow_soft_placement', default=True, help="Allow device soft device placement", action='store_true')
    parser.add_argument('--log_device_placement', default=False, help="Log placement of ops on devices", action='store_true')

    args, unparsed = parser.parse_known_args()
    config = args

    #DATASET_PATH = './'


    # model build
    # ==================================================
    vocabulary = []

    cnn = TextCNN(
        sequence_length=config.sequence_length,
        num_classes=1, # 여기서 class 개수를 수정해야 한다
        vocab_size=375230,
        embedding_size=config.embedding_dim,
        filter_sizes=list(map(int, config.filter_sizes.split(","))),
        num_filters=config.num_filters,
        l2_reg_lambda=config.l2_reg_lambda)

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(0.0005)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)


    ############ open sess
    session_conf = tf.ConfigProto(
        allow_soft_placement=config.allow_soft_placement,
        log_device_placement=config.log_device_placement)
    sess = tf.Session(config=session_conf)
    sess.run(tf.global_variables_initializer())


    ########### load data
    if config.mode == 'train':
        # data load
        print("Loading data...")
        data_path = DATASET_PATH
        x, y, vocabulary, vocabulary_inv = data_helpers.load_data(data_path)
        x_train = x
        y_train = y
        print("Vocabulary Size: {:d}".format(len(vocabulary)))
        #vocabulary = data_helpers.Vocab(voc=vocabulary)


    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config, model = cnn, vocabulary = vocabulary)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())


    ########### learning sess
    if config.mode == 'train':

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), config.batch_size, config.num_epochs)
        # Training loop. For each batch...
        saveiter = 0
        total_loss = 0
        iter = 0
        for batch in batches:
            iter += 1
            x_batch, y_batch = zip(*batch)
            loss = train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            total_loss += loss

            # DONOTCHANGE (You can decide how often you want to save the model)
            if current_step % config.checkpoint_every == 0:
                nsml.save(saveiter)
                print('saveiter %d: loss = %.4f' % (saveiter, total_loss / iter))
                saveiter += 1