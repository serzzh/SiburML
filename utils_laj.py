import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim



def dense_layer(x, size,activation_fn, batch_norm = False,phase=False, drop_out=False, keep_prob=None, scope="fc_layer"):
    """
    Helper function to create a fully connected layer with or without batch normalization or dropout regularization

    :param x: previous layer
    :param size: fully connected layer size
    :param activation_fn: activation function
    :param batch_norm: bool to set batch normalization
    :param phase: if batch normalization is set, then phase variable is to mention the 'training' and 'testing' phases
    :param drop_out: bool to set drop-out regularization
    :param keep_prob: if drop-out is set, then to mention the keep probability of dropout
    :param scope: variable scope name
    :return: fully connected layer
    """
    with tf.variable_scope(scope):
        if batch_norm:
            dence_layer = tf.contrib.layers.fully_connected(x, size, activation_fn=None)
            dence_layer_bn = BatchNorm(name="batch_norm_" + scope)(dence_layer, train=phase)
            return_layer = activation_fn(dence_layer_bn)
        else:
            return_layer = tf.layers.dense(x, size,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           activation=activation_fn)
        if drop_out:
            return_layer = tf.nn.dropout(return_layer, keep_prob)

        return return_layer


def get_RNNCell(cell_types, keep_prob, state_size, build_with_dropout=True):
    """
    Helper function to get a different types of RNN cells with or without dropout wrapper
    :param cell_types: cell_type can be 'GRU' or 'LSTM' or 'LSTM_LN' or 'GLSTMCell' or 'LSTM_BF' or 'None'
    :param keep_prob: dropout keeping probability
    :param state_size: number of cells in a layer
    :param build_with_dropout: to enable the dropout for rnn layers
    :return:
    """
    cells = []
    for cell_type in cell_types:
        if cell_type == 'GRU':
            cell = tf.contrib.rnn.GRUCell(num_units=state_size,
                                          bias_initializer=tf.zeros_initializer())  # Or GRU(num_units)
        elif cell_type == 'LSTM':
            cell = tf.contrib.rnn.LSTMCell(num_units=state_size, use_peepholes=True, state_is_tuple=True,
                                           initializer=tf.contrib.layers.xavier_initializer())
        elif cell_type == 'LSTM_LN':
            cell = tf.contrib.rnn.LayerNormBasicLSTMCell(state_size)
        elif cell_type == 'GLSTMCell':
            cell = tf.contrib.rnn.GLSTMCell(num_units=state_size, initializer=tf.contrib.layers.xavier_initializer())
        elif cell_type == 'LSTM_BF':
            cell = tf.contrib.rnn.LSTMBlockFusedCell(num_units=state_size, use_peephole=True)
        else:
            cell = tf.nn.rnn_cell.BasicRNNCell(state_size)

        if build_with_dropout:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        cells.append(cell)

    cell = tf.contrib.rnn.MultiRNNCell(cells)

    if build_with_dropout:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    return cell


class BatchNorm(object):
    """
    usage : dence_layer_bn = BatchNorm(name="batch_norm_" + scope)(previous_layer, train=is_train)
    """
    def __init__(self, epsilon=1e-5, momentum=0.999, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def batch_generator(x_train, y_train, batch_size, sequence_length, online=False, online_shift=1):
    """
    Generator function for creating sequential batches of training-data
    """
    num_x_sensors = x_train.shape[2]
    num_train = x_train.shape[0]
    idx = 0

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_sensors)
        x_batch = np.zeros(shape=x_shape, dtype=np.float32)
        #print("x_shape %s, idx %s" % (x_shape, idx))
        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length)
        y_batch = np.zeros(shape=y_shape, dtype=np.float32)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            
            x_batch[i] = x_train[idx+i]
            y_batch[i] = y_train[idx+i]
            #print(i,idx)
        if online:
            idx = idx + online_shift  # check if its nee to be idx=idx+1
        # print(idx)
        yield (x_batch, y_batch)

def model_summary(learning_rate,batch_size,lstm_layers,lstm_layer_size,fc_layer_size,sequence_length,n_channels,path_checkpoint,spacial_note=''):
    path_checkpoint=path_checkpoint + ".txt"
    if not os.path.exists(os.path.dirname(path_checkpoint)):
        os.makedirs(os.path.dirname(path_checkpoint))

    with open(path_checkpoint, "w") as text_file:
        variables = tf.trainable_variables()

        print('---------', file=text_file)
        print(path_checkpoint, file=text_file)
        print(spacial_note, file=text_file)
        print('---------', '\n', file=text_file)

        print('---------', file=text_file)
        #print('MAXLIFE: ', MAXLIFE,'\n',  file=text_file)
        print('learning_rate: ', learning_rate, file=text_file)
        print('batch_size: ', batch_size, file=text_file)
        print('lstm_layers: ', lstm_layers, file=text_file)
        print('lstm_layer_size: ', lstm_layer_size, file=text_file)
        print('fc_layer_size: ', fc_layer_size, '\n', file=text_file)
        print('sequence_length: ', sequence_length, file=text_file)
        print('n_channels: ', n_channels, file=text_file)
        print('---------', '\n', file=text_file)

        print('---------', file=text_file)
        print('Variables: name (type shape) [size]', file=text_file)
        print('---------', '\n', file=text_file)
        total_size = 0
        total_bytes = 0
        for var in variables:
            # if var.num_elements() is None or [] assume size 0.
            var_size = var.get_shape().num_elements() or 0
            var_bytes = var_size * var.dtype.size
            total_size += var_size
            total_bytes += var_bytes
            print(var.name, slim.model_analyzer.tensor_description(var), '[%d, bytes: %d]' %
                      (var_size, var_bytes), file=text_file)

        print('\nTotal size of variables: %d' % total_size, file=text_file)
        print('Total bytes of variables: %d' % total_bytes, file=text_file)


def scoring_func(error_arr):
    '''

    :param error_arr: a list of errors for each training trajectory
    :return: standered score value for RUL
    '''
    import math
    # print(error_arr)
    pos_error_arr = error_arr[error_arr >= 0]
    neg_error_arr = error_arr[error_arr < 0]

    score = 0
    # print(neg_error_arr)
    for error in neg_error_arr:
        score = math.exp(-(error / 13)) - 1 + score
        # print(math.exp(-(error / 13)),score,error)

    # print(pos_error_arr)
    for error in pos_error_arr:
        score = math.exp(error / 10) - 1 + score
        # print(math.exp(error / 10),score, error)
    return score


