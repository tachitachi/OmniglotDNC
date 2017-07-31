import argparse
import sys

from ops import *
from tardis import TardisCell, TardisStateTuple

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None

width = height = 28
num_classes = 10

def cnn():
    x = tf.placeholder(tf.float32, [None, width * height])
    obs = tf.reshape(x, [-1, width, height, 1])
    #x = tf.placeholder(tf.float32, [None, width, height, 1])
    
    # 3 conv layers
    #obs = batch_norm(conv2d(obs, 4, 'conv1', (3, 3), (2, 2)))
    #obs = batch_norm(conv2d(obs, 4, 'conv2', (3, 3), (2, 2)))
    #obs = batch_norm(conv2d(obs, 4, 'conv3', (3, 3), (2, 2)))
    
    obs = tf.nn.relu(conv2d(obs, 4, 'conv1', (3, 3), (2, 2)))
    obs = tf.nn.relu(conv2d(obs, 4, 'conv2', (3, 3), (2, 2)))
    #obs = tf.nn.relu(conv2d(obs, 4, 'conv3', (3, 3), (2, 2)))
    
    #obs = conv2d(obs, 4, 'conv1', (3, 3), (2, 2))
    #obs = conv2d(obs, 4, 'conv2', (3, 3), (2, 2))
    #obs = conv2d(obs, 4, 'conv3', (3, 3), (2, 2))
    #obs = conv2d(obs, 4, 'conv4', (3, 3), (2, 2))
    obs = flatten(obs)
    
    # linear layer
    #obs = linear(x, 100, 'linear_a')
    #obs = linear(obs, 50, 'linear0')
    #obs = linear(obs, 25, 'linear1')
    obs = linear(obs, num_classes, 'linear2')
    
    
    y = tf.placeholder(tf.float32, [None, num_classes])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=obs, labels=y))
    train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
    
    correct_prediction = tf.equal(tf.argmax(obs, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return x, y, loss, train_op, accuracy
    

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
    print(sequence.get_shape())
    
    seq_len = length_all(obs)
    #obs = tf.transpose(obs, [1, 0, 2])
    
    #obs = tf.split(obs, width, 0)
    #obs = tf.concat(obs, axis=0)
    #obs = tf.reshape(x, [-1, 1, width])
    
    print('1', type(obs), obs, tf.shape(obs)[:1])
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    
    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, obs, dtype=tf.float32, sequence_length=seq_len)
    #lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm, obs, dtype=tf.float32, sequence_length=l)
    
    print(lstm_outputs)
    
    obs = linear(flatten(lstm_outputs), num_classes, 'linear0')

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
    
    obs = linear(flatten(lstm_outputs), num_classes, 'linear0')

    y = tf.placeholder(tf.float32, [None, num_classes])
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=obs, labels=y))
    train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
    
    correct_prediction = tf.equal(tf.argmax(obs, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y, state_init, cell, loss, train_op, accuracy, seq_len, lstm_state

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    
    #x, y, loss, train_op, accuracy = cnn()
    x, y, state_init, cell, loss, train_op, accuracy, seq_len, lstm_state = rnn_tardis()
    
    batches = 5000
    batch_size = 200
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        for batch in range(batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #print(batch_xs, batch_ys)
            
            #state = cell.zero_state(len(batch_xs), dtype=tf.float32)
            #state = tf.nn.rnn_cell.LSTMStateTuple(np.zeros((batch_size, 64)), np.zeros((batch_size, 64)))
            
            
            #print('state', state)
            
            # train 
            _, z, sl = sess.run([train_op, loss, seq_len], feed_dict={x: batch_xs, y: batch_ys })
            #print(z)
                
            if batch % 100 == 0:
                acc, state = sess.run([accuracy, lstm_state], feed_dict={x: mnist.test.images, y: mnist.test.labels})
                #print(np.array([z, acc]), sl)
                print(np.array([z, acc]))
                #print(state)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)