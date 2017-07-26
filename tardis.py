import collections
import tensorflow as tf
import numpy as np
        
from ops import *
        
_TardisStateTuple = collections.namedtuple('TardisStateTuple', ('c', 'h', 'm'))

class TardisStateTuple(_TardisStateTuple):
    '''Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
    Stores two elements: `(c, h)`, in that order.
    Only used when `state_is_tuple=True`.
    '''
    __slots__ = ()

    @property
    def dtype(self):
        (c, h, m) = self
        if c.dtype != h.dtype or h.dtype != m.dtype:
            raise TypeError('Inconsistent internal state: %s, %s, and %s' %
                                            (str(c.dtype), str(h.dtype), str(m.dtype)))
        return c.dtype
        
        
        
        
        
class TardisCell(tf.nn.rnn_cell.RNNCell):
    """Tardis recurrent network cell with LSTM controller.
    The implementation is based on: https://arxiv.org/abs/1701.08718
    """

    def __init__(self, num_units, mem_size, word_size, forget_bias=1.0, activation=None, reuse=None):
        """Initialize the basic LSTM cell.
        Args:
            num_units: int, The number of units in the LSTM cell.
            forget_bias: float, The bias added to forget gates (see above).
            state_is_tuple: If True, accepted and returned states are 2-tuples of
                the `c_state` and `m_state`.    If False, they are concatenated
                along the column axis.    The latter behavior will soon be deprecated.
            activation: Activation function of the inner states.    Default: `tanh`.
            reuse: (optional) Python boolean describing whether to reuse variables
                in an existing scope.    If not `True`, and the existing scope already has
                the given variables, an error is raised.
        """
        super(TardisCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._mem_size = mem_size
        self._word_size = word_size
        self._forget_bias = forget_bias
        self._activation = activation or tf.tanh

    def zero_state(self, batch_size, dtype='float32'):
        c = np.zeros((batch_size, self._num_units), dtype=dtype)
        h = np.zeros((batch_size, self._num_units), dtype=dtype)
        m = np.zeros((batch_size, self._mem_size * self._word_size), dtype=dtype)
        
        return TardisStateTuple(c, h, m)
        
    @property
    def state_size(self):
        return TardisStateTuple(self._num_units, self._num_units, self._mem_size * self._word_size)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """LSTM with nunits cells and external memory."""
        print('creating tardis cell', state)
        print('inputs', inputs)

        c, h, m = state
        
        print(inputs.shape, h.shape, m.shape)
        
        #m_tiled = 

        # Read from memory
        # for this to work correctly, we need to feed in one at a time
        with tf.variable_scope('tardis_memory_read'):
            read_weights = _linear([inputs, h, m], self._mem_size, True)
            #read_weights = _linear([inputs, h], self._mem_size, True)
            
#            with tf.variable_scope('W_h'):
#                wh = _linear([h], self._word_size, False)
#                
#            with tf.variable_scope('W_x'):
#                wx = _linear([inputs], self._word_size, False)
#            
#            print(tf.shape(inputs)[0:1])
#            temp = tf.tile(m, tf.expand_dims(tf.shape(inputs)[0:1], 1))
#            print('@@@@@@@temp', temp)
#            m_tile = tf.reshape(temp, [-1, self._mem_size * self._word_size])
#            
#            print('m_tile', m_tile)
            
            #read_weights = _linear([inputs, h], self._mem_size, True)

            read_index = gumbel_softmax(read_weights, 1.0, hard=True)
            
            read_index = tf.reshape(tf.tile(read_index, (1, self._mem_size)),  [-1, self._mem_size, self._word_size])
            
            print('read_index', read_index)
            #print(tf.reshape(m, [-1, self._mem_size, self._word_size]))
            #print(tf.reshape(m, [-1, self._mem_size, self._word_size]) * tf.reshape(read_index, [-1, 1]))

            read_value = tf.reduce_sum(tf.reshape(m, [-1, self._mem_size, self._word_size]) * read_index, axis=1)

        print('@@@0', read_value)


        #concat = _linear([inputs, h, tf.expand_dims(read_value, 0)], 4 * self._num_units, True)
        with tf.variable_scope('tardis_reshape_test'):
            read_tiled = tf.reshape(tf.tile(read_value, [1, tf.shape(inputs)[0]]), [-1, self._word_size])
            
        print('read_tiled', read_tiled)
        
        #concat = _linear([inputs, h, read_value], 4 * self._num_units, True)
        concat = _linear([inputs, h], 4 * self._num_units, True)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

        new_c = (
            c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * tf.sigmoid(o)



        # Write to memory
        with tf.variable_scope('tardis_memory_write'):

            new_memory = _linear([new_h], self._word_size, True)
            print('new_memory', new_memory)

            erase = tf.reshape(tf.ones_like(read_index) - read_index, [-1, self._mem_size, self._word_size])

            print('@@@1', erase)

            #print('@@@2', tf.reshape(tf.tile(new_memory, [self._mem_size, 1]), [-1, self._mem_size, self._word_size]))
            #print('@@@3', tf.reshape(read_index, [-1, 1]))

            print(m)

            new_to_write = tf.reshape(tf.tile(new_memory, [self._mem_size, 1]), [-1, self._mem_size, self._word_size])
            #new_index = tf.reshape(read_index, [-1, 1])

            new_mem = new_to_write * read_index

            print(new_mem)

            m = tf.reshape(m, [-1, self._mem_size, self._word_size]) * erase + new_mem
            m = tf.reshape(m, [-1, self._mem_size * self._word_size])

            print('@@@4', m)

        return new_h, TardisStateTuple(new_c, new_h, m)


        
        

def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_initializer: starting value to initialize the bias
            (default is all zeros).
        kernel_initializer: starting value to initialize the weight.
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
#    if args is None or (nest.is_sequence(args) and not args):
#        raise ValueError("`args` must be specified")
#    if not nest.is_sequence(args):
#        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as outer_scope:
        weights = tf.get_variable(
                'kernel', [total_arg_size, output_size],
                dtype=dtype,
                initializer=kernel_initializer)
        if len(args) == 1:
            res = tf.matmul(args[0], weights)
        else:
            res = tf.matmul(tf.concat(args, 1), weights)
        if not bias:
            return res
        with tf.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = tf.constant_initializer(0.0, dtype=dtype)
            biases = tf.get_variable(
                    'bias', [output_size],
                    dtype=dtype,
                    initializer=bias_initializer)
        return tf.nn.bias_add(res, biases)