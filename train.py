import argparse
import sys
from omniglot import Omniglot
import tensorflow as tf
import numpy as np

from model import omniglot_rnn





FLAGS = None


def main(_):
    # TODO: Add data directory?
    og = Omniglot()
    

    
    #x, y, loss, train_op, accuracy, seq_len = rnn()
    
    batches = 50
    batch_size = 50
    num_classes = 5
    num_samples = 10
    
    
    
    x, y, loss, train_op, accuracy, seq_len = omniglot_rnn(num_classes)
    
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        for batch in range(batches):
            batch_xs, batch_ys = og.TrainBatch(batch_size, classes=num_classes, samples=num_samples, one_hot=True)
            
            shifted_ys = [np.zeros(num_classes)] + batch_ys[:-1]
            
            xs = np.concatenate((np.array(batch_xs), np.array(shifted_ys)), axis=1)
            ys = np.array(batch_ys)
            
            print(xs)
            
            #print(batch_xs)
            #print(batch_ys)
            #print(shifted_ys)
            
            _, l, acc, length = sess.run([train_op, loss, accuracy, seq_len], feed_dict={x: xs, y: ys})
            print(l, acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data_dir', type=str, default='./tmp/tensorflow/mnist/input_data',
    #                  help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)