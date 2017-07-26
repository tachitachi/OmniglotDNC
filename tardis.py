import collections
import tensorflow as tf
import numpy as np
        
        
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

    @property
    def state_size(self):
        return TardisStateTuple(self._num_units, self._num_units, self._mem_size * self._word_size)

    @property
    def output_size(self):
        return self._num_units

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM)."""
        
        
        c, h, m = state
        
        
        
        
        # Parameters of gates are concatenated into one multiply for efficiency.
        
        
        concat = _linear([inputs, h], 4 * self._num_units, True)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

        new_c = (
                c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * tf.sigmoid(o)
        
        
        
        
        

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