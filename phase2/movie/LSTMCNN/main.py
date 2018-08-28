# -*- coding: utf-8 -*-


import numpy as np
import os
import time
import datetime
import data_helpers_nsml as data_helpers
from lstm_cnn import LSTM_CNN  

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

        feed_dict = {
            model.input_x: preprocessed_data,
            model.dropout_keep_prob: 1.0
        }
        pred = sess.run(model.scores, feed_dict=feed_dict)
        pred = pred * 10
        pred = pred.flatten()
        con = np.zeros(len(pred))
        result = list(zip(con, pred))

        # DONOTCHANGE: They are reserved for nsml
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
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.dropout_keep_prob: config.dropout_keep_prob
        }

        _, step,  loss = sess.run(
            [train_op, global_step,  model.loss],
            feed_dict)
        accuracy = 0
        time_str = datetime.datetime.now().isoformat()

        return loss


    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            model.input_x: x_batch,
            model.input_y: y_batch,
            model.dropout_keep_prob: 1.0
        }

        step, devloss = sess.run([global_step, model.loss], feed_dict)
        accuracy = 0
        time_str = datetime.datetime.now().isoformat()
        print("validation step : {}: step {},\t validattion loss {:g},\t acc {:g}".format(time_str, step, devloss, accuracy))

        return devloss



    # Parameters
    # ==================================================

    parser = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--pause', type=int, default=0)
    parser.add_argument('--iteration', type=str, default='0')

    # Model Hyperparameters
    parser.add_argument('--filter_sizes', type=str, default="2,3,4",help="Comma-separated filter sizes (default: '3,4,5')")
    parser.add_argument('--embedding_dim', type=int, default=150, help="Dimensionality of character embedding (default: 128)")
    parser.add_argument('--num_filters', type=int, default=128, help="Number of filters per filter size (default: 128)")
    parser.add_argument('--hiddensize', type=int, default=150, help="Number of filters per filter size (default: 128)")
    parser.add_argument('--dropout_keep_prob', type=float, default=0.7, help="fc Dropout keep probability (default: 0.5)")
    parser.add_argument('--l2_reg_lambda', type=float, default=0, help="L2 regularizaion lambda (default: 0.0)")
    parser.add_argument('--lr', type=float, default=0.001, help="L2 regularizaion lambda (default: 0.0)")

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch Size (default: 64)") #1024
    parser.add_argument('--num_epochs', type=int, default=3, help="Number of training epochs (default: 200)")
    parser.add_argument('--evaluate_every', type=int, default=500, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument('--checkpoint_every', type=int, default=100, help="Save model after this many steps (default: 100)")    #100
    parser.add_argument('--dev_size', type=float, default=0.001, help="validation split size (default: 0.001)")

    # Misc Parameters
    parser.add_argument('--allow_soft_placement', default=True, help="Allow device soft device placement", action='store_true')
    parser.add_argument('--log_device_placement', default=False, help="Log placement of ops on devices", action='store_true')

    args, unparsed = parser.parse_known_args()
    config = args

    print(config)
    #DATASET_PATH = './'



    vocabulary = []


    ######################################################## load data
    print('{}'.format('#'*30))
    print('load data')
    if config.mode == 'train':
        # data load
        print("Loading data...")
        #data_path = './'
        data_path = DATASET_PATH
        x, y, vocabulary, vocabulary_inv = data_helpers.load_data(data_path)
        np.random.seed(42)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_sample_index = -1 * int(config.dev_size * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        print("Vocabulary Size: {:d}".format(len(vocabulary)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
        #vocabulary = data_helpers.Vocab(voc=vocabulary)

    ######################################################### model build
    print('{}'.format('#' * 30))
    print('model build')
    model = LSTM_CNN(sequence_length=8,
        num_classes=1, # 여기서 class 개수를 수정해야 한다
        vocab_size=803087,
        embedding_size=config.embedding_dim,
        filter_sizes=list(map(int, config.filter_sizes.split(","))),
        num_filters=config.num_filters,
        l2_reg_lambda=config.l2_reg_lambda,
                     num_hidden = config.hiddensize)
    

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(config.lr)
    grads_and_vars = optimizer.compute_gradients(model.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    print('{}'.format('#' * 30))
    print('sess open')
    sesstime = time.time()

    ############ open sess
    session_conf = tf.ConfigProto(
        allow_soft_placement=config.allow_soft_placement,
        log_device_placement=config.log_device_placement)
    sess = tf.Session(config=session_conf)
    sess.run(tf.global_variables_initializer())
    print('sess open time = {:.3f} sec'.format(time.time()-sesstime))


    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config, model = model, vocabulary = vocabulary)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())


    ########### learning sess
    print('{}'.format('#' * 30))
    print('train start')
    if config.mode == 'train':

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), config.batch_size, config.num_epochs)
        # Training loop. For each batch...
        saveiter = 0
        total_loss = 0
        iter = 0
        epochPerbatchnum = len(y_train) // config.batch_size + 1
        totalbatchnum = epochPerbatchnum * config.num_epochs
        minloss = 10000
        minloss_step = 0
        print('1 epoch is {} batchs'.format(epochPerbatchnum))
        for batch in batches:
            iter += 1
            x_batch, y_batch = zip(*batch)
            batch_loss = train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            total_loss += batch_loss

            # DONOTCHANGE (You can decide how often you want to save the model)
            rtime = time.time()
            if current_step % config.checkpoint_every == 0:
                savetime = time.time()
                nsml.save(saveiter)
                print('save time = {:.3f}'.format(time.time() - savetime))
                devtime = time.time()
                now_valloss = dev_step(x_dev, y_dev, writer=None)
                if now_valloss < minloss:
                    print('we find lower val_loss!!')
                    print('min saveiter : {} ----->>> {}'.format(minloss_step, saveiter))
                    minloss = now_valloss
                    minloss_step = saveiter
                #print('{}'.format('#'*30))
                print('epoch num = {} / {}'.format(iter//epochPerbatchnum, config.num_epochs))
                print('saveiter {}\tbatch  = {}/{} \ttrain loss = {:.4f}\t runtime = {:.3f} sec'.format(saveiter, iter, totalbatchnum, total_loss/iter, time.time() - rtime))
                print('now minimum vall_loss saveiter is {}\tval_loss = {:.3f}'.format(minloss_step, minloss))
                print('validattion time : {:.2f}\tval_data length : {}'.format(time.time() - devtime, len(y_dev)))
                print('{}'.format('#' * 30))
                rtime = 0
                saveiter += 1

