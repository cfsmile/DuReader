# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the Pointer Network for selecting answer spans, as described in:
https://openreview.net/pdf?id=B1-q5Pqxl
"""

import tensorflow as tf
import tensorflow.contrib as tc


def custom_dynamic_rnn(cell, inputs, inputs_len, initial_state=None):
    """
    Implements a dynamic rnn that can store scores in the pointer network,
    the reason why we implements this is that the raw_rnn or dynamic_rnn function in Tensorflow
    seem to require the hidden unit and memory unit has the same dimension, and we cannot
    store the scores directly in the hidden unit.
    Args:
        cell: RNN cell
        inputs: the input sequence to rnn
        inputs_len: valid length
        initial_state: initial_state of the cell
    Returns:
        outputs and state
    """
    batch_size = tf.shape(inputs)[0]
    max_time = tf.shape(inputs)[1]

    # tf.TensorArray is a class wrapping dynamic-sized, per-time-step,
    # write-once Tensor arrays.
    # This class is meant to be used with dynamic iteration primitives such
    # as while_loop and map_fn.  It supports gradient back-propagation via
    # special 'flow' control flow dependencies.
    inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
    # tf.TensorArray.unstack(value, name=None) unstacks the values of a
    # Tensor in the TensorArray.
    inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1, 0, 2]))
    emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
    t0 = tf.constant(0, dtype=tf.int32)
    if initial_state is not None:
        s0 = initial_state
    else:
        # zero_state is a method of Class AttentionCellWrapper
        # It is a basic attention cell wrapper.
        # Implementation based on https://arxiv.org/abs/1409.0473
        # It returns zero-flled state tensor(s).
        s0 = cell.zero_state(batch_size, dtype=tf.float32)
    f0 = tf.zeros([batch_size], dtype=tf.bool)

    def loop_fn(t, prev_s, emit_ta, finished):
        """
        the loop function of rnn
        """
        cur_x = inputs_ta.read(t)
        scores, cur_state = cell(cur_x, prev_s)
        # the 'cell' is a object constructor. The class seems to be
        # AttentionCellWrapper.

        # copy through
        # tf.where returns the elements, either from x or y, depending on the
        #  'condition'
        # The 'condition' tensor acts as a mask that chooses, based on the
        # value at each element, whether the corresponding element/row in the
        #  output should be taken from x(if true) or y(if false).
        scores = tf.where(finished,  # condition
                          tf.zeros_like(scores),  # x
                          scores  # y
                          )

        if isinstance(cell, tc.rnn.LSTMCell):
            cur_c, cur_h = cur_state
            prev_c, prev_h = prev_s
            cur_state = tc.rnn.LSTMStateTuple(tf.where(finished, prev_c, cur_c),
                                              tf.where(finished, prev_h, cur_h))
        else:
            cur_state = tf.where(finished, prev_s, cur_state)

        emit_ta = emit_ta.write(t, scores)
        finished = tf.greater_equal(t + 1, inputs_len)
        return [t + 1, cur_state, emit_ta, finished]

    # Repeat 'body' while the condition 'cond' is true
    # while_loop calls cond and body exactly once(inside the call to
    # while_loop, and not at all during Session.run().  while_loop stitches
    # together the graph fragments created during the cond and body calls
    # with some additional graph nodes to create the graph flow that repeats
    # body until cond returns false.
    # The output tensors for the loop variables after the loop.  When the
    # length of loop_vars is 1 this is a Tensor, TensorArray or IndexedSlice
    # and when the length of loop_vars is greater than 1 it returns a list.
    _, state, emit_ta, _ = tf.while_loop(
        # cond is a callable returning a boolean scalar tensor.
        cond=lambda _1, _2, _3, finished: tf.logical_not(tf.reduce_all(finished)),
        # tf.reduce_all computes the 'logical and' of elements across dimensions
        # tf.logical_not returns the truth value of NOT x element-wise.
        body=loop_fn,
        #  body is a callable returning a (possibly nested) tuple, namedtuple or
        #  list of tensors of the same arity(length and structure) and types
        #  as loop_vars
        loop_vars=(t0, s0, emit_ta, f0),
        # loop_vars is a (possibly nested) tuple, namedtuple or list of
        # tensors that is passed to both cond and body.  cond and body both
        # take as many arguments as there are loop_vars.
        parallel_iterations=32,
        swap_memory=False)

    outputs = tf.transpose(emit_ta.stack(), [1, 0, 2])
    return outputs, state


def attend_pooling(pooling_vectors, ref_vector, hidden_size, scope=None):
    """
    Applies attend pooling to a set of vectors according to a reference vector.
    Args:
        pooling_vectors: the vectors to pool
        ref_vector: the reference vector
        hidden_size: the hidden size for attention function
        scope: score name
    Returns:
        the pooled vector
    """

    # What does polling mean?
    with tf.variable_scope(scope or 'attend_pooling'):
        U = tf.tanh(tc.layers.fully_connected(pooling_vectors, num_outputs=hidden_size,
                                              activation_fn=None, biases_initializer=None)
                    + tc.layers.fully_connected(tf.expand_dims(ref_vector, 1),
                                                num_outputs=hidden_size,
                                                activation_fn=None))
        logits = tc.layers.fully_connected(U, num_outputs=1, activation_fn=None)
        scores = tf.nn.softmax(logits, 1)
        # reduce_sum computes the sum of elements across dimensions of a tensor.
        pooled_vector = tf.reduce_sum(pooling_vectors * scores, axis=1)
    return pooled_vector


class PointerNetLSTMCell(tc.rnn.LSTMCell):
    """
    Implements the Pointer Network Cell
    """
    def __init__(self, num_units, context_to_point):
        super(PointerNetLSTMCell, self).__init__(num_units, state_is_tuple=True)
        self.context_to_point = context_to_point
        self.fc_context = tc.layers.fully_connected(self.context_to_point,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)

    def __call__(self, inputs, state, scope=None):
        (c_prev, m_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            U = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(m_prev,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None),
                                         1))
            logits = tc.layers.fully_connected(U, num_outputs=1, activation_fn=None)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(self.context_to_point * scores, axis=1)
            lstm_out, lstm_state = super(PointerNetLSTMCell, self).__call__(attended_context, state)
        return tf.squeeze(scores, -1), lstm_state


class PointerNetDecoder(object):
    """
    Implements the Pointer Network
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def decode(self, passage_vectors, question_vectors, init_with_question=True):
        """
        Use Pointer Network to compute the probabilities of each position
        to be start and end of the answer
        Args:
            passage_vectors: the encoded passage vectors
            question_vectors: the encoded question vectors
            init_with_question: if set to be true,
                             we will use the question_vectors to init the state of Pointer Network
        Returns:
            the probs of evary position to be start and end of the answer
        """
        with tf.variable_scope('pn_decoder'):
            fake_inputs = tf.zeros([tf.shape(passage_vectors)[0], 2, 1])  # not used
            sequence_len = tf.tile([2], [tf.shape(passage_vectors)[0]])
            if init_with_question:
                # tf.random_normal outputs random values from anormal
                # distribution
                random_attn_vector = tf.Variable(tf.random_normal([1,
                                                                   self.hidden_size] # shape
                                                                  ),
                                                 trainable=True, name="random_attn_vector")
                # tf.contrib.layers.fully_connected adds a fully connected
                # layer.
                pooled_question_rep = tc.layers.fully_connected(
                    attend_pooling(question_vectors, random_attn_vector, self.hidden_size),
                    # inputs
                    num_outputs=self.hidden_size,
                    activation_fn=None
                )
                # Tuple used by LSTM Cells for state_size, zero_state,
                # and output state.  Stores two elements: (c, h), in that
                # order. Where c is the hidden state and h is the output.
                init_state = tc.rnn.LSTMStateTuple(pooled_question_rep, pooled_question_rep)
            else:
                init_state = None
            with tf.variable_scope('fw'):
                # PointerNetLSTMCell implements the Pointer Network Cell
                fw_cell = PointerNetLSTMCell(self.hidden_size, passage_vectors)
                fw_outputs, _ = custom_dynamic_rnn(fw_cell, fake_inputs, sequence_len, init_state)
            with tf.variable_scope('bw'):
                bw_cell = PointerNetLSTMCell(self.hidden_size, passage_vectors)
                bw_outputs, _ = custom_dynamic_rnn(bw_cell, fake_inputs, sequence_len, init_state)
            start_prob = (fw_outputs[0:, 0, 0:] + bw_outputs[0:, 1, 0:]) / 2
            end_prob = (fw_outputs[0:, 1, 0:] + bw_outputs[0:, 0, 0:]) / 2
            return start_prob, end_prob



