import tensorflow as tf
import numpy as np

def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad='SAME'):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        
        # does x have to be 4 dimensional?
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]
        
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        
        W = tf.get_variable('W', filter_shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(-w_bound, w_bound))
        b = tf.get_variable('b', [1, 1, 1, num_filters], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        
        return tf.nn.conv2d(x, W, stride_shape, pad) + b
        
def linear(x, size, name, initializer='relu', bias_init=0.0):
    with tf.variable_scope(name):
        if initializer == 'relu':
            fan_in = int(x.get_shape()[1])
            w_bound = tf.sqrt(2 / fan_in)
            initializer = tf.random_uniform_initializer(-w_bound, w_bound)
        W = tf.get_variable('W', [int(x.get_shape()[1]), size], initializer=initializer)
        b = tf.get_variable('b', [size], initializer=tf.constant_initializer(bias_init))
        
        return tf.matmul(x, W) + b
        
def batch_norm(x, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-7):
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon)
        
def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
    
def sample_gumbel(shape, eps=1e-20): 
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
