import argparse
import sys

from ops import *
from tardis import TardisCell, TardisStateTuple

from talstm import TimeAwareLSTMCell

from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imresize, imsave, imread

import tensorflow as tf
import numpy as np

FLAGS = None

width = height = 28
#width = 14
#height = 56

num_classes = 10

def cnn():

    epsilon = 0.15
    alpha = 0.5

    x = tf.placeholder(tf.float32, [None, height * width])


    def build_network(inputs, reuse=False):
        with tf.variable_scope('cnn_network', reuse=reuse):
            obs = tf.reshape(inputs, [-1, height, width, 1])
            #x = tf.placeholder(tf.float32, [None, width, height, 1])
            
            obs = tf.layers.conv2d(obs, 64, 3, strides=2, activation=tf.nn.relu, name='conv1')
            obs = tf.layers.conv2d(obs, 128, 3, strides=2, activation=tf.nn.relu, name='conv2')
            obs = tf.layers.conv2d(obs, 256, 3, strides=2, activation=tf.nn.relu, name='conv3')
            obs = flatten(obs)
            
            # linear layer
            obs = tf.layers.dense(obs, num_classes, name='linear2')

            return obs

    obs = build_network(x)
    
    
    y = tf.placeholder(tf.float32, [None, num_classes])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=obs, labels=y))

    grads = tf.gradients(loss, x)
    print('grads', grads)

    adv_x = tf.reshape(x + epsilon * tf.sign(grads), [-1, height * width])

    print('adv_obs', adv_x)
    print('x', x)

    # Adversarial training:
    # https://arxiv.org/pdf/1412.6572.pdf
    adv_obs = build_network(adv_x, reuse=True)
    adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=adv_obs, labels=y))

    total_loss = alpha * loss + (1 - alpha) * adv_loss




    #train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
    train_op = tf.train.AdamOptimizer(1e-2).minimize(total_loss)
    
    correct_prediction = tf.equal(tf.argmax(obs, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #return x, y, loss, train_op, accuracy
    return x, y, loss, train_op, accuracy, loss, adv_loss, adv_x, obs
    

def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length  
  
# Force everything to be max length
def length_all(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2) + 1)
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length  
    
# https://danijar.com/variable-sequence-lengths-in-tensorflow/
# http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
def rnn():

    rnn_size = 128
    
    x = tf.placeholder(tf.float32, [None, width * height])
    
    # batch_size, max_time, chunk_size
    sequence = obs = tf.reshape(x, [-1, width, height])

    # adding dt dimension to each item in the sequence
    ones = tf.ones([tf.shape(obs)[0], width, 1])
    sequence = obs = tf.concat([obs, ones], 2)

    print(sequence.get_shape())
    
    seq_len = length_all(obs)
    #obs = tf.transpose(obs, [1, 0, 2])
    
    #obs = tf.split(obs, width, 0)
    #obs = tf.concat(obs, axis=0)
    #obs = tf.reshape(x, [-1, 1, width])
    
    print('1', type(obs), obs, tf.shape(obs)[:1])
    lstm = TimeAwareLSTMCell(rnn_size)
    
    
    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, obs, dtype=tf.float32, sequence_length=seq_len)
    #lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, obs, dtype=tf.float32, sequence_length=l)
    
    print(lstm_outputs)
    
    obs = tf.layers.dense(flatten(lstm_outputs), num_classes, name='linear0')

    y = tf.placeholder(tf.float32, [None, num_classes])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=obs, labels=y))
    train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
    
    correct_prediction = tf.equal(tf.argmax(obs, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y, loss, train_op, accuracy, seq_len
    
    
def rnn_tardis():

    lstm_size = 64
    memory_size = 16
    word_size = 16
    
    x = tf.placeholder(tf.float32, [None, width * height])
    
    # batch_size, max_time, chunk_size
    sequence = obs = tf.reshape(x, [-1, width, height])
    print(sequence.get_shape())
    
    seq_len = length_all(obs)
    #obs = tf.transpose(obs, [1, 0, 2])
    
    #obs = tf.split(obs, width, 0)
    #obs = tf.concat(obs, axis=0)
    #obs = tf.reshape(x, [-1, 1, width])
    
    print('1', type(obs), obs, tf.shape(obs)[:1])
    
    useTardis = True
    
    
    if useTardis:
        cell = TardisCell(lstm_size, memory_size, word_size)
        
        c_placeholder = tf.placeholder(tf.float32, [None, cell.state_size.c])
        h_placeholder = tf.placeholder(tf.float32, [None, cell.state_size.h])
        m_placeholder = tf.placeholder(tf.float32, [None, cell.state_size.m])
        
        state_init = TardisStateTuple(c_placeholder, h_placeholder, m_placeholder)
    
    else:
        cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        
        c_placeholder = tf.placeholder(tf.float32, [None, cell.state_size.c])
        h_placeholder = tf.placeholder(tf.float32, [None, cell.state_size.h])
        
        state_init = tf.nn.rnn_cell.LSTMStateTuple(c_placeholder, h_placeholder)
    
    print('state_init', state_init)
    
    
    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        cell=cell, 
        inputs=obs, 
        #initial_state=state_init,
        dtype=tf.float32, 
        sequence_length=seq_len)
    #lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, obs, dtype=tf.float32, sequence_length=l)
    
    print(lstm_outputs)
    
    obs = tf.layers.dense(flatten(lstm_outputs), num_classes, name='linear0')

    y = tf.placeholder(tf.float32, [None, num_classes])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=obs, labels=y))
    train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
    
    correct_prediction = tf.equal(tf.argmax(obs, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y, state_init, cell, loss, train_op, accuracy, seq_len, lstm_state

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    
    x, y, loss, train_op, accuracy, partial_loss, adv_loss, adv_x, guess = cnn()
    #x, y, loss, train_op, accuracy = cnn()
    print('done creating cnn')
    #x, y, state_init, cell, loss, train_op, accuracy, seq_len, lstm_state = rnn_tardis()
    #x, y, loss, train_op, accuracy, seq_len = rnn()
    
    batches = 5000
    batch_size = 200
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        for batch in range(batches + 1):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #print(batch_xs, batch_ys)
            
            #state = cell.zero_state(len(batch_xs), dtype=tf.float32)
            #state = tf.nn.rnn_cell.LSTMStateTuple(np.zeros((batch_size, 64)), np.zeros((batch_size, 64)))
            
            
            #print('state', state)
            
            # train 
            _, z, p_loss, a_loss = sess.run([train_op, loss, partial_loss, adv_loss], feed_dict={x: batch_xs, y: batch_ys })
            #_, z = sess.run([train_op, loss], feed_dict={x: batch_xs, y: batch_ys })
            #print(z)
                
            if batch % 100 == 0:
                acc, adv_example = sess.run([accuracy, adv_x], feed_dict={x: mnist.test.images, y: mnist.test.labels})
                sample = np.reshape(mnist.test.images[0], (height, width))
                adv_sample = np.reshape(adv_example[0], (height, width))
                diff_sample = sample - adv_sample
                results = sess.run(guess, feed_dict={x: [mnist.test.images[0], adv_example[0]]})
                imsave('samples/sample_{}.png'.format(batch), sample)
                imsave('samples/sample_{}_adv.png'.format(batch), adv_sample)
                print(np.array([z, acc, p_loss, a_loss]), sample.shape, adv_sample.shape, diff_sample.shape, np.argmax(results, axis=1))
                #print(np.array([z, acc]))
                #print(state)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)