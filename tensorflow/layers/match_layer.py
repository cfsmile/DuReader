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
This module implements the core layer of Match-LSTM and BiDAF
"""

import tensorflow as tf
#  contrib module containing volatile or experimental code
#  tf.contrib.rnn, module: RNN Cells and addition RNN operations
import tensorflow.contrib as tc


class MatchLSTMAttnCell(tc.rnn.LSTMCell):
    #  LSTMCel inherits from LayerRNNCell. The class uses optional peep-hole
    # connections, optional cell clipping, and and an optional projection
    # layer.
    """
    Implements the Match-LSTM attention cell
    """
    def __init__(self, num_units, context_to_attend):
        # Call parent class LSTMCell __init__ to initialize the parameters
        # for an LSTM cell.
        super(MatchLSTMAttnCell, self).__init__(num_units,
                                                # int,
                                                # The number of units in the
                                                # LSTM cell.
                                                state_is_tuple=True
                                                # If True, accepted and
                                                # returned states are
                                                # 2-tuples of the c_state and
                                                #  m_state.
                                                )
        self.context_to_attend = context_to_attend
        self.fc_context = tc.layers.fully_connected(self.context_to_attend,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)

    def __call__(self, inputs, state, scope=None):
        (c_prev, h_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            ref_vector = tf.concat([inputs, h_prev], -1)
            G = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(ref_vector,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None), 1))
            logits = tc.layers.fully_connected(G, num_outputs=1, activation_fn=None)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(self.context_to_attend * scores, axis=1)
            new_inputs = tf.concat([inputs, attended_context,
                                    inputs - attended_context, inputs * attended_context],
                                   -1)
            return super(MatchLSTMAttnCell, self).__call__(new_inputs, state, scope)


class MatchLSTMLayer(object):
    """
    Implements the Match-LSTM layer, which attend to the question dynamically in a LSTM fashion.
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Match-LSTM algorithm
        """
        with tf.variable_scope('match_lstm'):
            cell_fw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            cell_bw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=passage_encodes,
                                                             sequence_length=p_length,
                                                             dtype=tf.float32)
            match_outputs = tf.concat(outputs, 2)
            state_fw, state_bw = state
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            match_state = tf.concat([h_fw, h_bw], 1)
        return match_outputs, match_state


class AttentionFlowMatchLayer(object):
    """
    Implements the Attention Flow layer,
    which computes Context-to-question Attention and question-to-context Attention
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Attention Flow Match algorithm
        """
        with tf.variable_scope('bidaf'):
            #  tf.matmul multiplies matrix a by matrix b, producing a*b
            sim_matrix = tf.matmul(passage_encodes,  # a
                                   question_encodes,  # b
                                   transpose_b=True
                                   # b is transposed before multiplication
                                   )
            # tf.nn.softmax computes softmax activations
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix,
                                                            # logits: A
                                                            # non-empty Tensor
                                                            -1
                                                            # axis: The
                                                            # dimension
                                                            # softmax would
                                                            # be performed
                                                            # on. The default
                                                            #  is -1 which
                                                            # indicates the
                                                            # last dimension.
                                                            ),  # a for matmul
                                              question_encodes  # b for matmul
                                              )
            # tf.expand_dims inserts a dimension of 1 into a tensor's shape
            # This ops is useful if you want to add a BATCH dimension to a
            # single element.
            # This ops is related to squeeze(), which removes dimensions of
            # size 1.

            # tf.reduce_max computes the maximum of elements across
            # dimensions of a tensor
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix,
                                                           # input_tensor for
                                                           #  reduce_max
                                                           2
                                                           # axis for
                                                           # reduce_max
                                                           ),
                                             # input for expand_dims
                                             1
                                             # axis for expand_dims
                                             ),
                              # logits for softmax
                              -1
                              # axis for softmax
                              )

            # tf.tile constructs a tensor by tiling a given tensor.  This ops
            #  creates a new tensor by replicating input 'multiples' times.
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                            # input
                                         [1, tf.shape(passage_encodes)[1], 1]
                                         # A tensor. Length must be the same
                                         # as the number of dimensions in input
                                            )
            # tf.concat concatenates tensors along one dimension.  That is,
            # the data from the input tensors is joined along the axis
            # dimension.
            # The number of dimensions of the input tensors must match,
            # and all dimensions except axis must be equal.
            concat_outputs = tf.concat([passage_encodes, context2question_attn,
                                        passage_encodes * context2question_attn,
                                        passage_encodes * question2context_attn],
                                       # values: a list of Tensor objects or
                                       # a single Tensor
                                       -1
                                       # axis
                                       )
            return concat_outputs, None
