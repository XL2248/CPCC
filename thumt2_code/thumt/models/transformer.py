# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy,code
import numpy as np
import tensorflow as tf
import thumt.layers as layers
import thumt.losses as losses
import thumt.utils as utils

from thumt.models.model import NMTModel


def _layer_process(x, mode, trainable=True):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x, trainable=trainable)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
               dtype=None, scope=None, trainable=True):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True, trainable=trainable)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True, trainable=trainable)

        return output

def _load_embedding(word_list, params, uniform_scale = 0.25, dimension_size = 300, embed_file='glove'):

    word2embed = {}
    if embed_file == 'w2v':
        file_path = params.embedding_path
    else:
        file_path = params.embedding_path

    with open(file_path, 'r') as fopen:
        for line in fopen:
            w = line.strip().split()
            word2embed[' '.join(w[:-dimension_size])] = w[-dimension_size:]
    word_vectors = []

    c = 0
    for word in word_list:
        if word in word2embed:
            c += 1
            s = np.array(word2embed[word], dtype=np.float32)
            word_vectors.append(s)
        else:
            word_vectors.append(np.random.uniform(-uniform_scale, uniform_scale, dimension_size))

    print('glove initializes {}'.format(c))
    print('all words initializes {}'.format(len(word_vectors)))

    return np.array(word_vectors, dtype=np.float32)

def birnn(inputs, sequence_length, params):
    lstm_fw_cell = rnn.BasicLSTMCell(params.hidden_size)
    lstm_bw_cell = rnn.BasicLSTMCell(params.hidden_size)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs,
                                                 sequence_length=sequence_length, dtype=tf.float32)
    states_fw, states_bw = outputs
    return tf.concat([states_fw, states_bw], axis=2)

def sample_gaussian(mu, logvar):
    epsilon = tf.random_normal(tf.shape(logvar), name="epsilon")
    std = tf.exp(0.5 * logvar)
    z= mu + tf.multiply(std, epsilon)
    return z

def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
                               - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
                               - tf.div(tf.exp(recog_logvar), tf.exp(prior_logvar)), reduction_indices=1)
    return kld

def gelu(input_tensor):
  """Gaussian Error Linear Unit.
  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    input_tensor: float Tensor to perform activation.
  Returns:
    `input_tensor` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf

def softplus(x):
    return np.log(1.0 + np.exp(x))

def w_encoder_attention(queries,
                        keys,
                        sequence_length,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        using_mask=False,
                        mymasks=None,
                        scope="w_encoder_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Set the fall back option for num_units
        # print(queries)
        # print(queries.get_shape().as_list)
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        # Linear projections

        Q = tf.layers.dense(queries, num_units, activation=None, use_bias=False)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None, use_bias=False)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None, use_bias=False)  # (N, T_k, C)

        x = K * Q
        x = tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1],num_heads, int(num_units/num_heads)])
        outputs = tf.transpose(tf.reduce_sum(x, 3),[0,2,1])
        outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)

        if using_mask:
            key_masks = mymasks
            key_masks = tf.reshape(tf.tile(key_masks, [1, num_heads]),
                                   [tf.shape(key_masks)[0], num_heads, tf.shape(key_masks)[1]])
        else:
            key_masks = tf.sequence_mask(sequence_length, tf.shape(keys)[1], dtype=tf.float32)
            key_masks = tf.reshape(tf.tile(key_masks,[1, num_heads]),[tf.shape(key_masks)[0],num_heads,tf.shape(key_masks)[1]])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
        outputs = tf.nn.softmax(outputs, 2)
        V_ = tf.reshape(V, [tf.shape(V)[0], tf.shape(V)[1], num_heads, int(num_units / num_heads)])
        V_ = tf.transpose(V_, [0, 2, 1, 3])
        outputs = tf.layers.dense(tf.reshape(tf.reduce_sum(V_ * tf.expand_dims(outputs, -1), 2), [-1, num_units]),
                                  num_units, activation=None, use_bias=False)
        weight = outputs
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
    return outputs, weight

def transformer_context(inputs, bias, params, dtype=None, scope="ctx_transformer", trainable=True):
    with tf.variable_scope(scope, default_name="context", dtype=dtype,
                           values=[inputs, bias], reuse=tf.AUTO_REUSE):
        x = inputs
        for layer in range(params.num_context_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        trainable=trainable
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                        trainable=trainable
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs

def transformer_encoder(inputs, bias, params, dia_mask=None, dtype=None, scope=None, trainable=True, get_first_layer=False):
    with tf.variable_scope("encoder", dtype=dtype,
                           values=[inputs, bias], reuse=tf.AUTO_REUSE):
        x = inputs
        for layer in range(params.num_encoder_layers):
            if layer < params.bottom_block:
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        max_relative_dis = params.max_relative_dis \
                            if params.position_info_type == 'relative' else None
    
                        y = layers.attention.multihead_attention(
                            _layer_process(x, params.layer_preprocess),
                            None,
                            bias,
                            params.num_heads,
                            params.attention_key_channels or params.hidden_size,
                            params.attention_value_channels or params.hidden_size,
                            params.hidden_size,
                            1.0 - params.attention_dropout,
                            max_relative_dis=max_relative_dis,
                            trainable=trainable
                        )
                        y = y["outputs"]
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess, trainable=trainable)

                    with tf.variable_scope("feed_forward"):
                        y = _ffn_layer(
                            _layer_process(x, params.layer_preprocess),
                            params.filter_size,
                            params.hidden_size,
                            1.0 - params.relu_dropout,
                            trainable=trainable
                        )
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess, trainable=trainable)
                first_layer_output = x
                print("first_layer_output", first_layer_output)
                if get_first_layer and layer == (params.bottom_block - 1):
                    return x, first_layer_output
            else:
                with tf.variable_scope("layer_%d" % layer):
                    with tf.variable_scope("self_attention"):
                        max_relative_dis = params.max_relative_dis \
                            if params.position_info_type == 'relative' else None

                        y = layers.attention.multihead_attention(
                            _layer_process(x, params.layer_preprocess),
                            None,
                            bias,
                            params.num_heads,
                            params.attention_key_channels or params.hidden_size,
                            params.attention_value_channels or params.hidden_size,
                            params.hidden_size,
                            1.0 - params.attention_dropout,
                            max_relative_dis=max_relative_dis,
                            trainable=trainable,
                            dia_mask=dia_mask
                        )
                        y = y["outputs"]
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess, trainable=trainable)

                    with tf.variable_scope("feed_forward"):
                        y = _ffn_layer(
                            _layer_process(x, params.layer_preprocess),
                            params.filter_size,
                            params.hidden_size,
                            1.0 - params.relu_dropout,
                            trainable=trainable
                        )
                        x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                        x = _layer_process(x, params.layer_postprocess, trainable=trainable)

#            if params.bottom_block and get_first_layer:
#                return first_layer_output, first_layer_output

        outputs = _layer_process(x, params.layer_preprocess)
        if params.bottom_block == 0:
            first_layer_output = x

        return outputs, first_layer_output


def transformer_decoder(inputs, memory, bias, mem_bias, params, state=None,
                        dtype=None, scope=None, trainable=True):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None
                max_relative_dis = params.max_relative_dis \
                        if params.position_info_type == 'relative' else None

                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        state=layer_state,
                        max_relative_dis=max_relative_dis,
                        trainable=trainable
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=trainable)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        max_relative_dis=max_relative_dis,
                        trainable=trainable
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=trainable)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                        trainable=trainable
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess, trainable=trainable)

        outputs = _layer_process(x, params.layer_preprocess)

        if state is not None:
            return outputs, next_state

        return outputs


def encoding_graph(features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    dtype = tf.get_variable_scope().dtype
    hidden_size = params.hidden_size
    src_seq = features["source"]
    print(features)
    ctx_dia_src_seq = features["context_dia_src"]
    ctx_dia_tgt_seq = features["context_dia_tgt"]
    ctx_sty_src_seq = features["context_sty_src"]
    ctx_sty_tgt_seq = features["context_sty_tgt"]
    ctx_lan_src_seq = features["context_lan_src"]
    ctx_lan_tgt_seq = features["context_lan_tgt"]


    #emotion = features["emotion"]
    src_len = features["source_length"]
    ctx_dia_src_len = features["context_dia_src_length"]
    ctx_dia_tgt_len = features["context_dia_tgt_length"]
    ctx_sty_src_len = features["context_sty_src_length"]
    ctx_sty_tgt_len = features["context_sty_tgt_length"]
    ctx_lan_src_len = features["context_lan_src_length"]
    ctx_lan_tgt_len = features["context_lan_tgt_length"]


    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)

    top_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["context_dia_src"])[1],
                                dtype=dtype or tf.float32)

    dia_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["context_dia_src"])[1],
                                dtype=dtype or tf.float32)

    true_mask = dia_mask * top_mask

    ctx_dia_src_mask = tf.sequence_mask(ctx_dia_src_len,
                                maxlen=tf.shape(features["context_dia_src"])[1],
                                dtype=tf.float32)
    ctx_dia_tgt_mask = tf.sequence_mask(ctx_dia_tgt_len,
                                maxlen=tf.shape(features["context_dia_tgt"])[1],
                                dtype=tf.float32)

    ctx_sty_src_mask = tf.sequence_mask(ctx_sty_src_len,
                                maxlen=tf.shape(features["context_sty_src"])[1],
                                dtype=tf.float32)
    ctx_sty_tgt_mask = tf.sequence_mask(ctx_sty_tgt_len,
                                maxlen=tf.shape(features["context_sty_tgt"])[1],
                                dtype=tf.float32)

    ctx_lan_src_mask = tf.sequence_mask(ctx_lan_src_len,
                                maxlen=tf.shape(features["context_lan_src"])[1],
                                dtype=tf.float32)
    ctx_lan_tgt_mask = tf.sequence_mask(ctx_lan_tgt_len,
                                maxlen=tf.shape(features["context_lan_tgt"])[1],
                                dtype=tf.float32)


    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer, trainable=True)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer, trainable=True)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer, trainable=True)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer, trainable=True)

    bias = tf.get_variable("bias", [hidden_size])

    inputs = tf.gather(src_embedding, src_seq)
    #emotion_inputs = tf.gather(src_embedding, emotion)

    if params.multiply_embedding_mode == "sqrt_depth":
        inputs = inputs * (hidden_size ** 0.5)
    #    emotion_inputs = emotion_inputs * (hidden_size ** 0.5)
    with tf.variable_scope("emotion_embedding"):
        if params.use_emovec:
#            emo_emb = tf.Variable(_load_embedding(params.vocabulary["emotion"], params), name="emo_embedding", trainable=False)
            emo_emb = tf.get_variable("emo_embedding", initializer=_load_embedding(params.vocabulary["emotion"], params), trainable=True)
        else:
            emo_emb = tf.get_variable("emo_embedding",
                                     [len(params.vocabulary["emotion"]), 300], initializer=tf.contrib.layers.xavier_initializer())
        emo_bias = tf.get_variable("emo_bias", [300])
        emo_inputs = tf.nn.embedding_lookup(emo_emb, features["emotion"])

    with tf.variable_scope("turn_position_embedding"):
        pos_emb = tf.get_variable("turn_pos_embedding", [len(params.vocabulary["position"]), hidden_size], initializer=tf.contrib.layers.xavier_initializer())

    inputs = inputs * tf.expand_dims(src_mask, -1) #src_mask, -1)

    encoder_input = tf.nn.bias_add(inputs, bias)
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    if params.position_info_type == 'absolute':
        encoder_input = layers.attention.add_timing_signal(encoder_input)
    #segment embeddings
    if params.segment_embeddings:
        seg_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_dia_src"])
        encoder_input += seg_pos_emb

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    #top_mask = tf.expand_dims(dia_mask, -1)
    encoder_output, first_layer_output = transformer_encoder(encoder_input, enc_attn_bias, params)
    ## context
    # ctx_seq: [batch, max_ctx_length]
    print("building context graph")
    if params.context_representation == "self_attention":
        print('use self attention')
        # dialogue src context
        get_first_layer = True
        dia_mask = None
        turn_dia_src_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_dia_src"])
        
        ctx_inputs = tf.gather(src_embedding, ctx_dia_src_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_dia_src_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_dia_src_pos_emb
        context_input = tf.nn.dropout(context_input, 1.0 - params.embed_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_dia_src_mask, "masking")
#        context_dia_src = transformer_context(context_input, ctx_attn_bias, params)
        context_dia_src, _ = transformer_encoder(context_input, ctx_attn_bias, params, dia_mask, get_first_layer)
        
        context_dia_src = first_layer_output

        # dialogue tgt context
        turn_dia_tgt_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_dia_tgt"])
        ctx_inputs = tf.gather(tgt_embedding, ctx_dia_tgt_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_dia_tgt_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_dia_tgt_pos_emb
        context_input = tf.nn.dropout(context_input, 1.0 - params.embed_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_dia_tgt_mask, "masking")
#        context_dia_tgt = transformer_context(context_input, ctx_attn_bias, params)
        context_dia_tgt, _ = transformer_encoder(context_input, ctx_attn_bias, params, dia_mask, get_first_layer)

        # style src context
        turn_sty_src_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_sty_src"])
        ctx_inputs = tf.gather(src_embedding, ctx_sty_src_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_sty_src_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_sty_src_pos_emb
        context_input = tf.nn.dropout(context_input, 1.0 - params.embed_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_sty_src_mask, "masking")
#        context_sty_src = transformer_context(context_input, ctx_attn_bias, params)
        context_sty_src, _ = transformer_encoder(context_input, ctx_attn_bias, params, dia_mask, get_first_layer)

        # style tgt context
        turn_sty_tgt_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_sty_tgt"])
        ctx_inputs = tf.gather(tgt_embedding, ctx_sty_tgt_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_sty_tgt_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_sty_tgt_pos_emb
        context_input = tf.nn.dropout(context_input, 1.0 - params.embed_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_sty_tgt_mask, "masking")
#        context_sty_tgt = transformer_context(context_input, ctx_attn_bias, params)
        context_sty_tgt, _ = transformer_encoder(context_input, ctx_attn_bias, params, dia_mask, get_first_layer)

        # language src context
        turn_lan_src_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_lan_src"])
        ctx_inputs = tf.gather(src_embedding, ctx_lan_src_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_lan_src_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_lan_src_pos_emb
        context_input = tf.nn.dropout(context_input, 1.0 - params.embed_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_lan_src_mask, "masking")
#        context_lan_src = transformer_context(context_input, ctx_attn_bias, params)
        context_lan_src, _ = transformer_encoder(context_input, ctx_attn_bias, params, dia_mask, get_first_layer)

        # language tgt context
        turn_lan_tgt_pos_emb = tf.nn.embedding_lookup(pos_emb, features["position_lan_tgt"])
        ctx_inputs = tf.gather(tgt_embedding, ctx_lan_tgt_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_lan_tgt_mask, -1)

        context_input = tf.nn.bias_add(ctx_inputs, bias)
        context_input = layers.attention.add_timing_signal(context_input)
        context_input = context_input + turn_lan_tgt_pos_emb
        context_input = tf.nn.dropout(context_input, 1.0 - params.embed_dropout)
        ctx_attn_bias = layers.attention.attention_bias(ctx_lan_tgt_mask, "masking")
#        context_lan_tgt = transformer_context(context_input, ctx_attn_bias, params)
        context_lan_tgt, _ = transformer_encoder(context_input, ctx_attn_bias, params, dia_mask, get_first_layer)
#        context_output = transformer_encoder(context_input, ctx_attn_bias, params)
    elif params.context_representation == "embedding":
        print('use embedding')
        ctx_inputs = tf.gather(src_embedding, ctx_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_mask, -1)
        context_input = tf.nn.bias_add(ctx_inputs, bias)
        ctx_attn_bias = layers.attention.attention_bias(ctx_mask, "masking")
        context_output = context_input
    elif params.context_representation == "bilstm":
        print('use bilstm')
        ctx_inputs = tf.gather(src_embedding, ctx_seq) * (hidden_size ** 0.5)
        ctx_inputs = ctx_inputs * tf.expand_dims(ctx_mask, -1)
        context_input = tf.nn.bias_add(ctx_inputs, bias)
        ctx_attn_bias = layers.attention.attention_bias(ctx_mask, "masking")
        context_output = birnn(context_input, ctx_len, params)

    return encoder_output, emo_inputs, context_dia_src, context_dia_tgt, context_sty_src, context_sty_tgt, context_lan_src, context_lan_tgt, turn_dia_src_pos_emb, turn_dia_tgt_pos_emb, turn_sty_src_pos_emb, turn_sty_tgt_pos_emb, turn_lan_src_pos_emb, turn_lan_tgt_pos_emb, first_layer_output


def decoding_graph(features, state, mode, params):
    is_training = True
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0
        is_training = False

    dtype = tf.get_variable_scope().dtype
    tgt_seq = features["target"]
    #src_len = features["context_dia_src_length"]    #
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=dtype or tf.float32)

    top_mask = tf.sequence_mask(features["source_length"],
                                maxlen=tf.shape(features["context_dia_src"])[1],
                                dtype=dtype or tf.float32)

    true_mask = src_mask * top_mask

    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=dtype or tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer, trainable=True)
    else:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("target_embedding",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer, trainable=True)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer, trainable=True)

    targets = tf.gather(tgt_embedding, tgt_seq)

    if params.multiply_embedding_mode == "sqrt_depth":
        targets = targets * (hidden_size ** 0.5)

    targets = targets * tf.expand_dims(tgt_mask, -1)

    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking",
                                                    dtype=dtype)
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal", dtype=dtype)
    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    if params.position_info_type == 'absolute':
        decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]
#    tgt_encoder_output = transformer_context(decoder_input, dec_attn_bias, params)
    tgt_encoder_output, first_layer_output_dec = transformer_encoder(decoder_input, dec_attn_bias, params)
    emo_inputs = state["emotion"]
    turn_dia_src_pos_emb = state["position_dia_src"]
    turn_dia_tgt_pos_emb = state["position_dia_tgt"]
    turn_sty_src_pos_emb = state["position_sty_src"]
    turn_sty_tgt_pos_emb = state["position_sty_tgt"]
    turn_lan_src_pos_emb = state["position_lan_src"]
    turn_lan_tgt_pos_emb = state["position_lan_tgt"]

    context_dia_src_output = state["context_dia_src"]
    context_dia_tgt_output = state["context_dia_tgt"]
    context_sty_src_output = state["context_sty_src"]
    context_sty_tgt_output = state["context_sty_tgt"]
    context_lan_src_output = state["context_lan_src"]
    context_lan_tgt_output = state["context_lan_tgt"]
    first_layer_output = state["first_layer_output"]

    emotion_ = emo_inputs[:,0,:]
    context_dia_src = context_dia_src_output[:,0,:]
#    context_dia_src = first_layer_output[:,0,:]
    context_dia_tgt = context_dia_tgt_output[:,0,:]
    context_sty_src = context_sty_src_output[:,0,:]
    context_sty_tgt = context_sty_tgt_output[:,0,:]
    context_lan_src = context_lan_src_output[:,0,:]
    context_lan_tgt = context_lan_tgt_output[:,0,:]
    
    s_mask = tf.expand_dims(src_mask, -1)
    t_mask = tf.expand_dims(tgt_mask, -1)

    src_rep = tf.reduce_sum(encoder_output * s_mask, -2) / tf.reduce_sum(s_mask, -2)

    # context latent
    if params.use_mtstyle_latent:
        ctx_prior = tf.layers.dense(tf.concat([src_rep, context_sty_src, context_sty_tgt], -1), hidden_size, activation=tf.nn.tanh, name="transform1")
        ctx_prior_mulogvar = tf.layers.dense(tf.layers.dense(ctx_prior, 256, activation=tf.nn.tanh), params.latent_dim * 2, use_bias=False, name="prior_fc1")
        ctx_prior_mu, ctx_prior_logvar = tf.split(ctx_prior_mulogvar, 2, axis=1)
        latent_sample_ctx = sample_gaussian(ctx_prior_mu, ctx_prior_logvar) # ctx
        latent_sample_ = latent_sample_ctx

    #emotion latent
    if params.use_dialog_latent:
        emo_prior = tf.layers.dense(tf.concat([src_rep, context_dia_src], -1), hidden_size, activation=tf.nn.tanh, name="transform2")
#        emo_prior = src_rep
        emo_prior_mulogvar = tf.layers.dense(tf.layers.dense(emo_prior, 256, activation=tf.nn.tanh), params.latent_dim * 2, use_bias=False, name="prior_fc2")
        emo_prior_mu, emo_prior_logvar = tf.split(emo_prior_mulogvar, 2, axis=1)
        latent_sample_emo = sample_gaussian(emo_prior_mu, emo_prior_logvar) # emo
        latent_sample_ = latent_sample_emo

    # third latent
    if params.use_language_latent:
        t_prior = tf.layers.dense(tf.concat([src_rep, context_lan_src, context_lan_tgt], -1), hidden_size, activation=tf.nn.tanh, name="transform3")
        prior_mulogvar3 = tf.layers.dense(tf.layers.dense(t_prior, 256, activation=tf.nn.tanh), params.latent_dim * 2, use_bias=False, name="prior_fc3")
        prior_mu3, prior_logvar3 = tf.split(prior_mulogvar3, 2, axis=1)
        latent_sample_3 = sample_gaussian(prior_mu3, prior_logvar3) # language
        latent_sample_ = latent_sample_3

    if mode != "infer":
        decoder_output = transformer_decoder(decoder_input, encoder_output,
                                             dec_attn_bias, enc_attn_bias,
                                             params)
        tgt_rep = tf.reduce_sum(tgt_encoder_output * t_mask, -2) / tf.reduce_sum(t_mask, -2)
        # post context latent
        if params.use_mtstyle_latent:
            ctx_post = tf.concat([src_rep, context_sty_src, context_sty_tgt, tgt_rep], axis=-1)
            ctx_post_encode = tf.layers.dense(ctx_post, hidden_size, use_bias=False, name="mixedencoderdecoder_sty")
            post_mulogvar_ctx = tf.layers.dense(ctx_post_encode, params.latent_dim * 2, use_bias=False, name="post_fc1")
            ctx_post_mu, ctx_post_logvar = tf.split(post_mulogvar_ctx, 2, axis=1)
            latent_sample_ctx_p = sample_gaussian(ctx_post_mu, ctx_post_logvar)
            latent_sample_ = latent_sample_ctx_p

        if params.use_dialog_latent:
        # post emotion latent
            emo_post = tf.concat([src_rep, context_dia_src, tgt_rep], axis=-1)
            emo_post_encode = tf.layers.dense(emo_post, hidden_size, use_bias=False, name="mixedencoderdecoder_dia")
            post_mulogvar_emo = tf.layers.dense(emo_post_encode, params.latent_dim * 2, use_bias=False, name="post_fc2")
            emo_post_mu, emo_post_logvar = tf.split(post_mulogvar_emo, 2, axis=1)
            latent_sample_emo_p = sample_gaussian(emo_post_mu, emo_post_logvar)
            latent_sample_ = latent_sample_emo_p

        if params.use_language_latent:
        # three latent
            t_post = tf.concat([src_rep, context_lan_src, context_lan_tgt, tgt_rep], axis=-1)
            t_post_encode = tf.layers.dense(t_post, hidden_size, use_bias=False, name="mixedencoderdecoder_lan")
            post_mulogvar_t = tf.layers.dense(t_post_encode, params.latent_dim * 2, use_bias=False, name="post_fc")
            t_post_mu, t_post_logvar = tf.split(post_mulogvar_t, 2, axis=1)
            latent_sample_t_p = sample_gaussian(t_post_mu, t_post_logvar)
            latent_sample_ = latent_sample_t_p

        if params.use_dialog_latent and params.use_language_latent and params.use_mtstyle_latent:
            latent_sample_ = tf.layers.dense(tf.concat([latent_sample_ctx_p, latent_sample_emo_p, latent_sample_t_p], axis=-1), params.latent_dim, use_bias=False, name="latent_fc1")
        #print("last",latent_sample_)
    else:
        # latent_sample_ = sample_gaussian(prior_mu, prior_logvar)
        if params.use_dialog_latent and params.use_language_latent and params.use_mtstyle_latent:
            latent_sample_ = tf.layers.dense(tf.concat([latent_sample_ctx, latent_sample_emo, latent_sample_3], axis=-1), params.latent_dim, use_bias=False, name="latent_fc1")
        #latent_sample_ = tf.concat([latent_sample_ctx, latent_sample_emo], axis=-1)

        decoder_input = decoder_input[:, -1:, :]
        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias,
                                              params, state=state["decoder"])

        decoder_output, decoder_state = decoder_outputs
        decoder_output = decoder_output[:, -1, :]
        if params.use_dialog_latent or params.use_language_latent or params.use_mtstyle_latent:
            out_lat = tf.concat([decoder_output, latent_sample_], axis=-1)
            out_lat1 = tf.layers.dense(out_lat, params.hidden_size, activation=tf.tanh, use_bias=False, name="last")
        else:
            out_lat1 = decoder_output
        logits = tf.matmul(out_lat1, weights, False, True)

        #logits = tf.matmul(decoder_output, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"encoder": encoder_output, "decoder": decoder_state, "emotion": emo_inputs, "context_dia_src": context_dia_src_output, "context_dia_tgt": context_dia_tgt_output, "context_sty_src": context_sty_src_output, "context_sty_tgt": context_sty_tgt_output, "context_lan_src": context_lan_src_output, "context_lan_tgt": context_lan_tgt_output, "position_dia_src": turn_dia_src_pos_emb, "position_dia_tgt": turn_dia_tgt_pos_emb, "position_sty_src": turn_sty_src_pos_emb, "position_sty_tgt": turn_sty_tgt_pos_emb, "position_lan_src": turn_lan_src_pos_emb, "position_lan_tgt": turn_lan_tgt_pos_emb, "first_layer_output": first_layer_output}

    if params.use_dialog_latent or params.use_language_latent or params.use_mtstyle_latent:
        latent_sample = tf.tile(tf.expand_dims(latent_sample_, 1), [1, tf.shape(decoder_output)[-2], 1])
    #code.interact(local=locals())
        out_lat = tf.concat([decoder_output, latent_sample], axis=-1)
        out_lat1 = tf.layers.dense(out_lat, hidden_size, activation=tf.tanh, use_bias=False, name="last")
    else:
        out_lat1 = decoder_output
    decoder_output1 = tf.reshape(out_lat1, [-1, hidden_size])

#    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output1, weights, False, True)
    labels = features["target"]

    # label smoothing
    ce = losses.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )
    tgt_mask = tf.cast(tgt_mask, ce.dtype)

    ce = tf.reshape(ce, tf.shape(tgt_seq))

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    ce_loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    kl_loss = 0.0
    if params.use_dialog_latent:
        kld1 = gaussian_kld(emo_post_mu, emo_post_logvar, emo_prior_mu, emo_prior_logvar)
        kl_loss1 = tf.reduce_mean(kld1)
        kl_loss = kl_loss1

    if params.use_mtstyle_latent:
        kld2 = gaussian_kld(ctx_post_mu, ctx_post_logvar, ctx_prior_mu, ctx_prior_logvar)
        kl_loss2 = tf.reduce_mean(kld2)
        kl_loss = kl_loss2

    if params.use_language_latent:
        kld3 = gaussian_kld(t_post_mu, t_post_logvar, prior_mu3, prior_logvar3)
        kl_loss3 = tf.reduce_mean(kld3)
        kl_loss = kl_loss3

    if params.use_dialog_latent and params.use_language_latent and params.use_mtstyle_latent:
        kl_loss = kl_loss1 + kl_loss2 + kl_loss3
#    kl_loss = tf.reduce_mean(kld) #* kl_weights
    #code.interact(local=locals())
    # bow loss
    avg_bow_loss = 0
    if params.use_bowloss:
        weights_bow = tf.get_variable("softmax_bow", [tgt_vocab_size, params.latent_dim + hidden_size],
                                  initializer=initializer)
        src_latent = tf.concat([latent_sample_ctx, tgt_rep], axis=-1)
        bow_logits = tf.matmul(src_latent, weights_bow, False, True)
        tile_bow_logits = tf.tile(tf.expand_dims(bow_logits, 1), [1, tf.shape(features["target"])[1], 1])
        bow_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tile_bow_logits, labels=labels) * tgt_mask
        bow_loss = tf.reduce_sum(bow_loss, reduction_indices=1)
        avg_bow_loss  = tf.reduce_mean(bow_loss)
        #bow = losses.smoothed_sigmoid_cross_entropy_with_logits(logits=bow_logits, labels=labels)
        #bow_loss = tf.reduce_sum(bow)
        #code.interact(local=locals())
    return ce_loss, kl_loss, avg_bow_loss


def model_graph(features, mode, params):
    encoder_output, emo_inputs, context_dia_src, context_dia_tgt, context_sty_src, context_sty_tgt, context_lan_src, context_lan_tgt, turn_dia_src_pos_emb, turn_dia_tgt_pos_emb, turn_sty_src_pos_emb, turn_sty_tgt_pos_emb, turn_lan_src_pos_emb, turn_lan_tgt_pos_emb,first_layer_output = encoding_graph(features, mode, params)
    state = {
        "encoder": encoder_output,
        "emotion": emo_inputs,
        "context_dia_src": context_dia_src,
        "context_dia_tgt": context_dia_tgt,
        "context_sty_src": context_sty_src,
        "context_sty_tgt": context_sty_tgt,
        "context_lan_src": context_lan_src,
        "context_lan_tgt": context_lan_tgt,
        "position_dia_src": turn_dia_src_pos_emb,
        "position_dia_tgt": turn_dia_tgt_pos_emb,
        "position_sty_src": turn_sty_src_pos_emb,
        "position_sty_tgt": turn_sty_tgt_pos_emb,
        "position_lan_src": turn_lan_src_pos_emb,
        "position_lan_tgt": turn_lan_tgt_pos_emb,
        "first_layer_output": first_layer_output
    }
    output = decoding_graph(features, state, mode, params)

    return output


class Transformer(NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer, regularizer=None, dtype=None):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            custom_getter = utils.custom_getter if dtype else None

            with tf.variable_scope(self._scope, initializer=initializer,
                                   regularizer=regularizer, reuse=reuse,
                                   custom_getter=custom_getter, dtype=dtype):
                loss = model_graph(features, "train", params)
                return loss#, kl_loss, bow_loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score, _ = model_graph(features, "eval", params)

            return score

        return evaluation_fn

    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output, emo_inputs, context_dia_src, context_dia_tgt, context_sty_src, context_sty_tgt, context_lan_src, context_lan_tgt, turn_dia_src_pos_emb, turn_dia_tgt_pos_emb, turn_sty_src_pos_emb, turn_sty_tgt_pos_emb, turn_lan_src_pos_emb, turn_lan_tgt_pos_emb, first_layer_output = encoding_graph(features, "infer", params)
                batch = tf.shape(encoder_output)[0]

                state = {
                    "encoder": encoder_output,
                    "emotion": emo_inputs,
                    "context_dia_src": context_dia_src,
                    "context_dia_tgt": context_dia_tgt,
                    "context_sty_src": context_sty_src,
                    "context_sty_tgt": context_sty_tgt,
                    "context_lan_src": context_lan_src,
                    "context_lan_tgt": context_lan_tgt,
                    "position_dia_src": turn_dia_src_pos_emb,
                    "position_dia_tgt": turn_dia_tgt_pos_emb,
                    "position_sty_src": turn_sty_src_pos_emb,
                    "position_sty_tgt": turn_sty_tgt_pos_emb,
                    "position_lan_src": turn_lan_src_pos_emb,
                    "position_lan_tgt": turn_lan_tgt_pos_emb,
                    "first_layer_output": first_layer_output,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.attention_key_channels or params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.attention_value_channels or params.hidden_size])
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }
            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            num_units=512,
            use_bowloss=False,
            use_srcctx=True,
            use_dialog_latent=True,
            use_language_latent=True,
            use_mtstyle_latent=True,
            use_emovec=True,
            segment_embeddings=False,
            hidden_size=512,
            latent_dim=32,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.5,
            relu_dropout=0.0,
            embed_dropout=0.1,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            context_representation="self_attention",
            num_context_layers=1,
            bottom_block=1,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0,
            # "absolute" or "relative"
            position_info_type="relative",
            # 8 for big model, 16 for base model, see (Shaw et al., 2018)
            max_relative_dis=16
        )

        return params
