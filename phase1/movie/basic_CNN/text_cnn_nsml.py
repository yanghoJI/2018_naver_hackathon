# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            init_range1 = math.sqrt(6.0 / (vocab_size+embedding_size))
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], minval=-init_range1, maxval=init_range1, seed=201706211703),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                init_range2 = math.sqrt(6.0 / sum(filter_shape))
                W = tf.Variable(tf.random_uniform(filter_shape, minval=-init_range2, maxval=init_range2, seed=201706211624), name="W")
                #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):


            ###여기 L2로 봐꾸면 될꺼야 알았지?
            #losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            losses = tf.losses.mean_squared_error(self.input_y, self.scores)
            print('y = {}'.format(self.input_y * 0.1))
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            a=1
            #correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            #self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            #self.accuracy = 0

            # top3 accuracy 계산을 위해 추가한 부분이다 정답이 1등, 2등, 3등에 있는 확률을 or 연산으로 더해준다
            #is_top = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            #is_2nd = tf.equal(tf.nn.top_k(self.scores, k=2)[1][:,1], tf.cast(tf.argmax(self.input_y, 1), "int32"))
            #is_3rd = tf.equal(tf.nn.top_k(self.scores, k=3)[1][:,2], tf.cast(tf.argmax(self.input_y, 1), "int32"))
            #is_in_top = is_top
            #is_in_top2 = tf.logical_or(is_in_top, is_2nd)
            #is_in_top3 = tf.logical_or(is_in_top2, is_3rd)
            #self.accuracy_top3 = tf.reduce_mean(tf.cast(is_in_top3, "float"), name="accuracy")


