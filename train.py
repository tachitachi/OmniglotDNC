import argparse
import sys
from omniglot import Omniglot
import tensorflow as tf
import numpy as np

from tardis import TardisStateTuple, TardisCell

from model import omniglot_rnn





FLAGS = None


def main(_):
    # TODO: Add data directory?
    og = Omniglot()
    

    
    #x, y, loss, train_op, accuracy, seq_len = rnn()
    
    batches = 25000
    batch_size = 20
    num_classes = 5
    num_samples = 5
    
    
    # Clean this up by turning into a class
    x, y, loss, train_op, accuracy, seq_len, zero_state, state_init, tardis_state, cell, guess, obs, softmax, lstm_init = omniglot_rnn(num_classes)
    
    
    for var in tf.global_variables():
        print(var)
    
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        for batch in range(batches):
            batch_xs, batch_ys = og.TrainBatch(batch_size, classes=num_classes, samples=num_samples, one_hot=True)
            
            shifted_ys = [np.zeros(num_classes)] + batch_ys[:-1]
            
            xs = np.concatenate((np.array(batch_xs), np.array(shifted_ys)), axis=1)
            #xs = np.concatenate((np.array(batch_xs), np.array(batch_ys)), axis=1)
            ys = np.array(batch_ys)
            
            #print(batch)
            
            #state = zero_state
            #state = cell.zero_state(batch_size)
            
            #state = cell.zero_state(1)
            #state = tf.nn.rnn_cell.LSTMStateTuple(np.zeros((1, 64), np.float32), np.zeros((1, 64), np.float32))
            #print('start')
            #for i in range(xs.shape[0]):
            #    currX = xs[i]
            #    currY = ys[i]
            #    g, state, logits = sess.run([guess, tardis_state, obs], feed_dict={x: [currX], y: [currY], lstm_init: state})
            #    print(g, currY)
            #    print(logits)
            #    #print(state)
            #print('end')
            #print(batch_xs)
            #print(batch_ys)
            #print(shifted_ys)
            
            state = cell.zero_state(1)
            #state = tf.nn.rnn_cell.LSTMStateTuple(np.zeros((1, 64), np.float32), np.zeros((1, 64), np.float32))
            
            _, l, acc, g, sl, out_state, logits, stoch = sess.run([train_op, loss, accuracy, guess, seq_len, tardis_state, obs, softmax], feed_dict={x: xs, y: ys, state_init: state})
            if batch % 250 == 0:
                print(l, acc, sl)
                print(g)
                print(np.argmax(batch_ys, 1))
                print(logits)
                
            if batch % 2000 == 0:
                print(out_state)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data_dir', type=str, default='./tmp/tensorflow/mnist/input_data',
    #                  help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)