# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math, code
import operator

import numpy as np
import tensorflow as tf
import thumt.utils.distribute as distribute


def batch_examples(example, batch_size, max_length, mantissa_bits,
                   shard_multiplier=1, length_multiplier=1, constant=False,
                   num_threads=4, drop_long_sequences=True):
    """ Batch examples

    :param example: A dictionary of <feature name, Tensor>.
    :param batch_size: The number of tokens or sentences in a batch
    :param max_length: The maximum length of a example to keep
    :param mantissa_bits: An integer
    :param shard_multiplier: an integer increasing the batch_size to suit
        splitting across data shards.
    :param length_multiplier: an integer multiplier that is used to
        increase the batch sizes and sequence length tolerance.
    :param constant: Whether to use constant batch size
    :param num_threads: Number of threads
    :param drop_long_sequences: Whether to drop long sequences

    :returns: A dictionary of batched examples
    """

    with tf.name_scope("batch_examples"):
        max_length = max_length or batch_size
        min_length = 8
        mantissa_bits = mantissa_bits

        # Compute boundaries
        x = min_length
        boundaries = []

        while x < max_length:
            boundaries.append(x)
            x += 2 ** max(0, int(math.log(x, 2)) - mantissa_bits)

        # Whether the batch size is constant
        if not constant:
            batch_sizes = [max(1, batch_size // length)
                           for length in boundaries + [max_length]]
            batch_sizes = [b * shard_multiplier for b in batch_sizes]
            bucket_capacities = [2 * b for b in batch_sizes]
        else:
            batch_sizes = batch_size * shard_multiplier
            bucket_capacities = [2 * n for n in boundaries + [max_length]]

        max_length *= length_multiplier
        boundaries = [boundary * length_multiplier for boundary in boundaries]
        max_length = max_length if drop_long_sequences else 10 ** 9

        # The queue to bucket on will be chosen based on maximum length
        max_example_length = 0
        for v in example.values():
            if v.shape.ndims > 0:
                seq_length = tf.shape(v)[0]
                max_example_length = tf.maximum(max_example_length, seq_length)

        (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
            max_example_length,
            example,
            batch_sizes,
            [b + 1 for b in boundaries],
            num_threads=num_threads,
            capacity=2,  # Number of full batches to store, we don't need many.
            bucket_capacities=bucket_capacities,
            dynamic_pad=True,
            keep_input=(max_example_length <= max_length)
        )

    return outputs

def get_turn_position(file1):
    with open(file1, 'r', encoding='utf-8') as fr:
        content = fr.readlines()
    turn_position = []
    for line in content:
        tmp = []
        index = 1
        for i in line.split()[::-1]:
            #tmp.append(str(index))
            if i == '[SEP]':
                index += 1
            tmp.append(str(index))
        if len(line.split()) != len(tmp):
            print(line)
        turn_position.append(tmp)
    base_path = '/'.join(file1.split('/')[:-1])
    signal = file1.split('/')[-1] #.split('.')[0]
    position_file = base_path + '/' + signal + '.turn_position'
    with open(position_file, 'w', encoding='utf-8') as fw:
        for line_position in turn_position:
            line_position = sorted(line_position, reverse=True)
            fw.write(' '.join(line_position) + '\n')
    #code.interact(local=locals())
    return position_file

def get_turn_position_src(file1, file2):
    with open(file1, 'r', encoding='utf-8') as fr:
        sentence = fr.readlines()
    with open(file2, 'r', encoding='utf-8') as fr:
        content = fr.readlines()

    turn_position = []
    mask = []
    for src, ctx in zip(sentence, content):
        line = ctx.replace('\n', ' ') + src.replace('\n', ' ') + '<eos> '
        tmp = []
        mask_sent = []
        index = 1
        flag = 0
        for i in line.split()[::-1]:
            if flag == 0:
                mask_sent.append(str(1))
                tmp.append(str(0))
            else:
                #mask_sent.append(str(0))
                tmp.append(str(index))
                if i == '[SEP]':
                    index += 1

            if i == '<eos>':
                flag = 1

        if len(line.split()) != len(tmp):
            print(line)
        turn_position.append(tmp)
        mask.append(mask_sent) 

    base_path = '/'.join(file1.split('/')[:-1])
    signal = file1.split('/')[-1] #.split('.')[0]
    position_file = base_path + '/' + signal + '.src_dia_turn_position'
    mask_file = base_path + '/' + signal + '.src_dia.mask'

    with open(position_file, 'w', encoding='utf-8') as fw:
        for line_position in turn_position:
            line_position = sorted(line_position, reverse=True)
            fw.write(' '.join(line_position) + '\n')

    with open(mask_file, 'w', encoding='utf-8') as fw:
        for sub_mask in mask:
            fw.write(' '.join(sub_mask) + '\n')

    #code.interact(local=locals())
    return position_file, mask_file

def get_training_input_contextual(filenames, filename_emo, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filenames, target_filenames]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """

    with tf.device("/cpu:0"):
        src_dataset = tf.data.TextLineDataset(filenames[0])
        tgt_dataset = tf.data.TextLineDataset(filenames[1])
        #ctx_dataset = tf.data.TextLineDataset(filename_ctx)
        emo_dataset = tf.data.TextLineDataset(filename_emo)
        ctx_dia_src_dataset = tf.data.TextLineDataset(params.dialog_src_context)
        ctx_dia_tgt_dataset = tf.data.TextLineDataset(params.dialog_tgt_context)
        ctx_sty_src_dataset = tf.data.TextLineDataset(params.style_src_context)
        ctx_sty_tgt_dataset = tf.data.TextLineDataset(params.style_tgt_context)
        ctx_lan_src_dataset = tf.data.TextLineDataset(params.language_src_context)
        ctx_lan_tgt_dataset = tf.data.TextLineDataset(params.language_tgt_context)

        position_file_src_dia = get_turn_position(params.dialog_src_context)
#        position_file_src_dia, mask_file = get_turn_position_src(filenames[0], params.dialog_src_context)
        position_file_tgt_dia = get_turn_position(params.dialog_tgt_context)
        position_file_src_sty = get_turn_position(params.style_src_context)
        position_file_tgt_sty = get_turn_position(params.style_tgt_context)
        position_file_src_lan = get_turn_position(params.language_src_context)
        position_file_tgt_lan = get_turn_position(params.language_tgt_context)

        position_src_dia_dataset = tf.data.TextLineDataset(position_file_src_dia)
        #mask_dataset = tf.data.TextLineDataset(mask_file)
        position_tgt_dia_dataset = tf.data.TextLineDataset(position_file_tgt_dia)
        position_src_sty_dataset = tf.data.TextLineDataset(position_file_src_sty)
        position_tgt_sty_dataset = tf.data.TextLineDataset(position_file_tgt_sty)
        position_src_lan_dataset = tf.data.TextLineDataset(position_file_src_lan)
        position_tgt_lan_dataset = tf.data.TextLineDataset(position_file_tgt_lan)
#        code.interact(local=locals())
        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, emo_dataset, ctx_dia_src_dataset, ctx_dia_tgt_dataset, ctx_sty_src_dataset, ctx_sty_tgt_dataset, ctx_lan_src_dataset, ctx_lan_tgt_dataset, position_src_dia_dataset, position_tgt_dia_dataset, position_src_sty_dataset, position_tgt_sty_dataset, position_src_lan_dataset, position_tgt_lan_dataset))

        if distribute.is_distributed_training_mode():
            dataset = dataset.shard(distribute.size(), distribute.rank())

        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()

        # Split string
        dataset = dataset.map(
            lambda src, tgt, emo, ctx_dia_src, ctx_dia_tgt, ctx_sty_src, ctx_sty_tgt, ctx_lan_src, ctx_lan_tgt, pos_dia_src, pos_dia_tgt, pos_sty_src, pos_sty_tgt, pos_lan_src, pos_lan_tgt: (
                tf.string_split([src]).values,
                tf.string_split([tgt]).values,
                tf.string_split([emo]).values,
                tf.string_split([ctx_dia_src]).values,
                tf.string_split([ctx_dia_tgt]).values,
                tf.string_split([ctx_sty_src]).values,
                tf.string_split([ctx_sty_tgt]).values,
                tf.string_split([ctx_lan_src]).values,
                tf.string_split([ctx_lan_tgt]).values,
                tf.string_split([pos_dia_src]).values,
                tf.string_split([pos_dia_tgt]).values,
                tf.string_split([pos_sty_src]).values,
                tf.string_split([pos_sty_tgt]).values,
                tf.string_split([pos_lan_src]).values,
                tf.string_split([pos_lan_tgt]).values
            ),
            num_parallel_calls=params.num_threads
        )

        # Append <eos> symbol
        dataset = dataset.map(
            lambda src, tgt, emo, ctx_dia_src, ctx_dia_tgt, ctx_sty_src, ctx_sty_tgt, ctx_lan_src, ctx_lan_tgt, pos_dia_src, pos_dia_tgt, pos_sty_src, pos_sty_tgt, pos_lan_src, pos_lan_tgt: (
                tf.concat([src, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt, [tf.constant(params.eos)]], axis=0),
                emo,
                ctx_dia_src,#tf.concat([src, [tf.constant(params.eos)], ctx_dia_src], axis=0),
                ctx_dia_tgt, 
                ctx_sty_src,
                ctx_sty_tgt,
                ctx_lan_src,
                ctx_lan_tgt, 
                pos_dia_src, 
                pos_dia_tgt, 
                pos_sty_src, 
                pos_sty_tgt, 
                pos_lan_src, 
                pos_lan_tgt
            ),
            num_parallel_calls=params.num_threads
        )

        # Convert to dictionary
        dataset = dataset.map(
            lambda src, tgt, emo, ctx_dia_src, ctx_dia_tgt, ctx_sty_src, ctx_sty_tgt, ctx_lan_src, ctx_lan_tgt, pos_dia_src, pos_dia_tgt, pos_sty_src, pos_sty_tgt, pos_lan_src, pos_lan_tgt: {
                "source": src,
                "target": tgt,
                "emotion": emo,
                "context_dia_src": ctx_dia_src,
                "context_dia_tgt": ctx_dia_tgt,
                "context_sty_src": ctx_sty_src,
                "context_sty_tgt": ctx_sty_tgt,
                "context_lan_src": ctx_lan_src,
                "context_lan_tgt": ctx_lan_tgt,
                "position_dia_src": pos_dia_src,
                "position_dia_tgt": pos_dia_tgt,
                "position_sty_src": pos_sty_src,
                "position_sty_tgt": pos_sty_tgt,
                "position_lan_src": pos_lan_src,
                "position_lan_tgt": pos_lan_tgt,
                "source_length": tf.shape(src),
                "target_length": tf.shape(tgt),
                "context_dia_src_length": tf.shape(ctx_dia_src),
                "context_dia_tgt_length": tf.shape(ctx_dia_tgt),
                "context_sty_src_length": tf.shape(ctx_sty_src),
                "context_sty_tgt_length": tf.shape(ctx_sty_tgt),
                "context_lan_src_length": tf.shape(ctx_lan_src),
                "context_lan_tgt_length": tf.shape(ctx_lan_tgt)
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )
        emo_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["emotion"]),
            default_value=-1
        )
        pos_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["position"]),
            default_value=-1
        )

        # String to index lookup
        features["source"] = src_table.lookup(features["source"])
        features["target"] = tgt_table.lookup(features["target"])
        features["context_dia_src"] = src_table.lookup(features["context_dia_src"])
        features["context_sty_src"] = src_table.lookup(features["context_sty_src"])
        features["context_lan_src"] = src_table.lookup(features["context_lan_src"])

        features["context_dia_tgt"] = tgt_table.lookup(features["context_dia_tgt"])
        features["context_sty_tgt"] = tgt_table.lookup(features["context_sty_tgt"])
        features["context_lan_tgt"] = tgt_table.lookup(features["context_lan_tgt"])

        features["emotion"] = emo_table.lookup(features["emotion"])
        features["position_dia_src"] = pos_table.lookup(features["position_dia_src"])
        features["position_dia_tgt"] = pos_table.lookup(features["position_dia_tgt"])
        features["position_sty_src"] = pos_table.lookup(features["position_sty_src"])
        features["position_sty_tgt"] = pos_table.lookup(features["position_sty_tgt"])
        features["position_lan_src"] = pos_table.lookup(features["position_lan_src"])
        features["position_lan_tgt"] = pos_table.lookup(features["position_lan_tgt"])

        # Batching
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=len(params.device_list),
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Convert to int32
        features["source"] = tf.to_int32(features["source"])
        features["target"] = tf.to_int32(features["target"])
        features["source_length"] = tf.to_int32(features["source_length"])
        features["target_length"] = tf.to_int32(features["target_length"])
        features["source_length"] = tf.squeeze(features["source_length"], 1)
        features["target_length"] = tf.squeeze(features["target_length"], 1)

        features["context_dia_src"] = tf.to_int32(features["context_dia_src"])
        features["context_dia_tgt"] = tf.to_int32(features["context_dia_tgt"])
        features["context_sty_src"] = tf.to_int32(features["context_sty_src"])
        features["context_sty_tgt"] = tf.to_int32(features["context_sty_tgt"])
        features["context_lan_src"] = tf.to_int32(features["context_lan_src"])
        features["context_lan_tgt"] = tf.to_int32(features["context_lan_tgt"])

        features["emotion"] = tf.to_int32(features["emotion"])
        features["position_dia_src"] = tf.to_int32(features["position_dia_src"])
        features["position_dia_tgt"] = tf.to_int32(features["position_dia_tgt"])
        features["position_sty_src"] = tf.to_int32(features["position_sty_src"])
        features["position_sty_tgt"] = tf.to_int32(features["position_sty_tgt"])
        features["position_lan_src"] = tf.to_int32(features["position_lan_src"])
        features["position_lan_tgt"] = tf.to_int32(features["position_lan_tgt"])

        features["context_dia_src_length"] = tf.to_int32(features["context_dia_src_length"])
        features["context_dia_src_length"] = tf.squeeze(features["context_dia_src_length"], 1)
        features["context_dia_tgt_length"] = tf.to_int32(features["context_dia_tgt_length"])
        features["context_dia_tgt_length"] = tf.squeeze(features["context_dia_tgt_length"], 1)

        features["context_sty_src_length"] = tf.to_int32(features["context_sty_src_length"])
        features["context_sty_src_length"] = tf.squeeze(features["context_sty_src_length"], 1)
        features["context_sty_tgt_length"] = tf.to_int32(features["context_sty_tgt_length"])
        features["context_sty_tgt_length"] = tf.squeeze(features["context_sty_tgt_length"], 1)

        features["context_lan_src_length"] = tf.to_int32(features["context_lan_src_length"])
        features["context_lan_src_length"] = tf.squeeze(features["context_lan_src_length"], 1)
        features["context_lan_tgt_length"] = tf.to_int32(features["context_lan_tgt_length"])
        features["context_lan_tgt_length"] = tf.squeeze(features["context_lan_tgt_length"], 1)

        return features

def get_training_input_contextual_emo(filenames, params):

    """ Get input for training stage

    :param filenames: A list contains [source_filenames, target_filenames]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """
    #print("filenames:", filenames)
    with tf.device("/cpu:0"):

        datasets = []
        #code.interact(local=locals())
        for data in filenames:# bianli 4 ge file
            dataset = tf.data.Dataset.from_tensor_slices(data)
            # Split string
            dataset = dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=params.num_threads)
            # Append <eos>
            dataset = dataset.map(
                lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
                num_parallel_calls=params.num_threads
            )
            datasets.append(dataset)
        #code.interact(local=locals())
        dataset = tf.data.Dataset.zip(tuple(datasets))
        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()
        #for one_element in tfe.Iterator(dataset):
        #    print(one_element)
        # Convert to dictionary
        dataset = dataset.map(
            lambda *x: {
                "source": x[0],
                "target": x[1],
                "context": x[2],
                "emotion": x[3],
                "source_length": tf.shape(x[0]),
                "target_length": tf.shape(x[1]),
                "context_length": tf.shape(x[2])
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )

        # String to index lookup
        features["source"] = src_table.lookup(features["source"])
        features["target"] = tgt_table.lookup(features["target"])
        features["context"] = src_table.lookup(features["context"])
        features["emotion"] = src_table.lookup(features["emotion"])

        # Batching
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=len(params.device_list),
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Convert to int32
        features["source"] = tf.to_int32(features["source"])
        features["target"] = tf.to_int32(features["target"])
        features["context"] = tf.to_int32(features["context"])
        features["emotion"] = tf.to_int32(features["emotion"])

        features["source_length"] = tf.to_int32(features["source_length"])
        features["target_length"] = tf.to_int32(features["target_length"])
        features["context_length"] = tf.to_int32(features["context_length"])
        features["source_length"] = tf.squeeze(features["source_length"], 1)
        features["target_length"] = tf.squeeze(features["target_length"], 1)
        features["context_length"] = tf.squeeze(features["context_length"], 1)

        return features

def get_training_input(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filenames, target_filenames]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """

    with tf.device("/cpu:0"):
        src_dataset = tf.data.TextLineDataset(filenames[0])
        tgt_dataset = tf.data.TextLineDataset(filenames[1])

        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

        if distribute.is_distributed_training_mode():
            dataset = dataset.shard(distribute.size(), distribute.rank())

        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()

        # Split string
        dataset = dataset.map(
            lambda src, tgt: (
                tf.string_split([src]).values,
                tf.string_split([tgt]).values
            ),
            num_parallel_calls=params.num_threads
        )

        # Append <eos> symbol
        dataset = dataset.map(
            lambda src, tgt: (
                tf.concat([src, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt, [tf.constant(params.eos)]], axis=0)
            ),
            num_parallel_calls=params.num_threads
        )

        # Convert to dictionary
        dataset = dataset.map(
            lambda src, tgt: {
                "source": src,
                "target": tgt,
                "source_length": tf.shape(src),
                "target_length": tf.shape(tgt)
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )

        # String to index lookup
        features["source"] = src_table.lookup(features["source"])
        features["target"] = tgt_table.lookup(features["target"])

        # Batching
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=len(params.device_list),
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Convert to int32
        features["source"] = tf.to_int32(features["source"])
        features["target"] = tf.to_int32(features["target"])
        features["source_length"] = tf.to_int32(features["source_length"])
        features["target_length"] = tf.to_int32(features["target_length"])
        features["source_length"] = tf.squeeze(features["source_length"], 1)
        features["target_length"] = tf.squeeze(features["target_length"], 1)

        return features


def sort_input_file(filename, reverse=True):
    # Read file
    with tf.gfile.Open(filename) as fd:
        inputs = [line.strip() for line in fd]

    input_lens = [
        (i, len(line.strip().split())) for i, line in enumerate(inputs)
    ]

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i

    return sorted_keys, sorted_inputs

def sort_input_file_ctx(filename, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, reverse=True):
    # Read file
    with tf.gfile.Open(filename) as fd:
        inputs = [line.strip() for line in fd]

 #   with tf.gfile.Open(filename_ctx) as fd:
 #       ctxs = [line.strip() for line in fd]

    with tf.gfile.Open(f1) as fd:
        ctx1 = [line.strip() for line in fd]
    with tf.gfile.Open(f2) as fd:
        ctx2 = [line.strip() for line in fd]
    with tf.gfile.Open(f3) as fd:
        ctx3 = [line.strip() for line in fd]
    with tf.gfile.Open(f4) as fd:
        ctx4 = [line.strip() for line in fd]
    with tf.gfile.Open(f5) as fd:
        ctx5 = [line.strip() for line in fd]
    with tf.gfile.Open(f6) as fd:
        ctx6 = [line.strip() for line in fd]
    with tf.gfile.Open(f7) as fd:
        ctx7 = [line.strip() for line in fd]
    with tf.gfile.Open(f8) as fd:
        ctx8 = [line.strip() for line in fd]
    with tf.gfile.Open(f9) as fd:
        ctx9 = [line.strip() for line in fd]
    with tf.gfile.Open(f10) as fd:
        ctx10 = [line.strip() for line in fd]
    with tf.gfile.Open(f11) as fd:
        ctx11 = [line.strip() for line in fd]
    with tf.gfile.Open(f12) as fd:
        ctx12 = [line.strip() for line in fd]
    with tf.gfile.Open(f13) as fd:
        ctx13 = [line.strip() for line in fd]

#    return inputs,ctx1,ctx2,ctx3,ctx4,ctx5,ctx6,ctx7,ctx8,ctx9,ctx10,ctx11,ctx12,ctx13
    input_lens = [
        (i, len(line.strip().split())) for i, line in enumerate(inputs)
    ]

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=reverse)
    sorted_keys = {}
    sorted_inputs = []
    sorted_ctxs = []
    dialog_src_context, pos_src_dia, dialog_tgt_context, pos_tgt_dia, style_src_context, pos_src_sty, style_tgt_context, pos_tgt_sty, language_src_context, pos_src_lan, language_tgt_context, pos_tgt_lan, emo = [], [], [], [],[], [],[], [],[], [],[], [],[]

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
#        sorted_ctxs.append(ctxs[index])

        dialog_src_context.append(ctx1[index])
        pos_src_dia.append(ctx2[index])
        dialog_tgt_context.append(ctx3[index])
        pos_tgt_dia.append(ctx4[index])
        style_src_context.append(ctx5[index])
        pos_src_sty.append(ctx6[index])
        style_tgt_context.append(ctx7[index])
        pos_tgt_sty.append(ctx8[index])
        language_src_context.append(ctx9[index])
        pos_src_lan.append(ctx10[index])
        language_tgt_context.append(ctx11[index])
        pos_tgt_lan.append(ctx12[index])
        emo.append(ctx13[index])

        sorted_keys[index] = i

    return sorted_keys, sorted_inputs, dialog_src_context, pos_src_dia, dialog_tgt_context, pos_tgt_dia, style_src_context, pos_src_sty, style_tgt_context, pos_tgt_sty, language_src_context, pos_src_lan, language_tgt_context, pos_tgt_lan, emo

def sort_and_zip_files(names):
    inputs = []
    input_lens = []
    files = [tf.gfile.GFile(name) for name in names]

    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines]
        input_lens.append((count, len(lines[0].split())))
        inputs.append(lines)
        count += 1

    # Close files
    for fd in files:
        fd.close()

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=True)
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
    #a = []
    #for x in zip(*sorted_inputs):
    #    a.append(list(x))
    #code.interact(local=locals())
    return [list(x) for x in zip(*sorted_inputs)]

def get_evaluation_input_ctx(inputs, params):
    with tf.device("/cpu:0"):
        # Create datasets
        datasets = []
        print(len(inputs))
        for i, data in enumerate(inputs):
            dataset = tf.data.Dataset.from_tensor_slices(data)
            # Split string
            dataset = dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=params.num_threads)
            # Append <eos>
            if i > 0 and i < 13:
                dataset = dataset.map(
                    lambda x: x,
                    num_parallel_calls=params.num_threads
                )
            else:
                dataset = dataset.map(
                    lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
                    num_parallel_calls=params.num_threads
                )

            datasets.append(dataset)

        dataset = tf.data.Dataset.zip(tuple(datasets))

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda *x: {
                "source": x[0],
                "source_length": tf.shape(x[0])[0],
                "context_dia_src": x[1],
                "position_dia_src": x[2],
                "context_dia_src_length": tf.shape(x[1])[0],
                "context_dia_tgt": x[3],
                "position_dia_tgt": x[4],
                "context_dia_tgt_length": tf.shape(x[3])[0],
                "context_sty_src": x[5],
                "position_sty_src": x[6],
                "context_sty_src_length": tf.shape(x[5])[0],
                "context_sty_tgt": x[7],
                "position_sty_tgt": x[8],
                "context_sty_tgt_length": tf.shape(x[7])[0],
                "context_lan_src": x[9],
                "position_lan_src": x[10],
                "context_lan_src_length": tf.shape(x[9])[0],
                "context_lan_tgt": x[11],
                "position_lan_tgt": x[12],
                "context_lan_tgt_length": tf.shape(x[11])[0],
                "emotion": x[13],
                "references": x[14:]
            },
            num_parallel_calls=params.num_threads
        )
#        code.interact(local=locals())
        dataset = dataset.padded_batch(
            params.eval_batch_size,
            {
                "source": [tf.Dimension(None)],
                "source_length": [],
                "context_dia_src": [tf.Dimension(None)],
                "context_dia_tgt": [tf.Dimension(None)],
                "context_sty_src": [tf.Dimension(None)],
                "context_sty_tgt": [tf.Dimension(None)],
                "context_lan_src": [tf.Dimension(None)],
                "context_lan_tgt": [tf.Dimension(None)],
                "context_dia_src_length": [],
                "context_dia_tgt_length": [],
                "context_sty_src_length": [],
                "context_sty_tgt_length": [],
                "context_lan_src_length": [],
                "context_lan_tgt_length": [],
                "position_dia_src":  [tf.Dimension(None)],
                "position_dia_tgt":  [tf.Dimension(None)],
                "position_sty_src":  [tf.Dimension(None)],
                "position_sty_tgt":  [tf.Dimension(None)],
                "position_lan_src":  [tf.Dimension(None)],
                "position_lan_tgt":  [tf.Dimension(None)],
                "emotion": [tf.Dimension(None)],
                "references": (tf.Dimension(None),) * (len(inputs) - 14)
            },
            {
                "source": params.pad,
                "source_length": 0,
                "context_dia_src": params.pad,
                "context_dia_tgt": params.pad,
                "context_sty_src": params.pad,
                "context_sty_tgt": params.pad,
                "context_lan_src": params.pad,
                "context_lan_tgt": params.pad,
                "context_dia_src_length": 0,
                "context_dia_tgt_length": 0,
                "context_sty_src_length": 0,
                "context_sty_tgt_length": 0,
                "context_lan_src_length": 0,
                "context_lan_tgt_length": 0,
                "position_dia_src": params.pad,
                "position_dia_tgt": params.pad,
                "position_sty_src": params.pad,
                "position_sty_tgt": params.pad,
                "position_lan_src": params.pad,
                "position_lan_tgt": params.pad,
                "emotion": params.pad,
                "references": (params.pad,) * (len(inputs) - 14)
            }
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Covert source symbols to ids
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )
        emo_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["emotion"]),
            default_value=-1
        )
        pos_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["position"]),
            default_value=-1
        )
        features["source"] = src_table.lookup(features["source"])
        features["emotion"] = emo_table.lookup(features["emotion"])
        features["context_dia_src"] = src_table.lookup(features["context_dia_src"])
        features["context_dia_tgt"] = tgt_table.lookup(features["context_dia_tgt"])
        features["context_sty_src"] = src_table.lookup(features["context_sty_src"])
        features["context_sty_tgt"] = tgt_table.lookup(features["context_sty_tgt"])
        features["context_lan_src"] = src_table.lookup(features["context_lan_src"])
        features["context_lan_tgt"] = tgt_table.lookup(features["context_lan_tgt"])
        features["position_dia_src"] = pos_table.lookup(features["position_dia_src"])
        features["position_dia_tgt"] = pos_table.lookup(features["position_dia_tgt"])
        features["position_sty_src"] = pos_table.lookup(features["position_sty_src"])
        features["position_sty_tgt"] = pos_table.lookup(features["position_sty_tgt"])
        features["position_lan_src"] = pos_table.lookup(features["position_lan_src"])
        features["position_lan_tgt"] = pos_table.lookup(features["position_lan_tgt"])

    return features

def get_evaluation_input(inputs, params):
    with tf.device("/cpu:0"):
        # Create datasets
        datasets = []

        for data in inputs:
            dataset = tf.data.Dataset.from_tensor_slices(data)
            # Split string
            dataset = dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=params.num_threads)
            # Append <eos>
            dataset = dataset.map(
                lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
                num_parallel_calls=params.num_threads
            )
            datasets.append(dataset)

        dataset = tf.data.Dataset.zip(tuple(datasets))

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda *x: {
                "source": x[0],
                "source_length": tf.shape(x[0])[0],
                "references": x[1:]
            },
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            params.eval_batch_size,
            {
                "source": [tf.Dimension(None)],
                "source_length": [],
                "references": (tf.Dimension(None),) * (len(inputs) - 1)
            },
            {
                "source": params.pad,
                "source_length": 0,
                "references": (params.pad,) * (len(inputs) - 1)
            }
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Covert source symbols to ids
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )

        features["source"] = src_table.lookup(features["source"])

    return features

def get_inference_input_bak(inputs, params):
    if params.generate_samples:
        batch_size = params.sample_batch_size
    else:
        batch_size = params.decode_batch_size

    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(inputs)
        )

        # Split string
        dataset = dataset.map(lambda x: tf.string_split([x]).values,
                              num_parallel_calls=params.num_threads)

        # Append <eos>
        dataset = dataset.map(
            lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
            num_parallel_calls=params.num_threads
        )

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda x: {"source": x, "source_length": tf.shape(x)[0]},
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            batch_size * len(params.device_list),
            {"source": [tf.Dimension(None)], "source_length": []},
            {"source": params.pad, "source_length": 0}
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        features["source"] = src_table.lookup(features["source"])

        return features

def get_inference_input(inputs, data_type, params):

    with tf.device("/cpu:0"):
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(inputs)
        )

        # Split string
        dataset = dataset.map(lambda x: tf.string_split([x]).values,
                              num_parallel_calls=params.num_threads)

        # Append <eos>
        if data_type == "x":
            print("input x:", data_type)
            dataset = dataset.map(
                lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
                num_parallel_calls=params.num_threads
            )
        else:
            dataset = dataset.map(
                lambda x: x,
                num_parallel_calls=params.num_threads
            )

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda x: {"source": x, "source_length": tf.shape(x)[0]},
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            params.decode_batch_size * len(params.device_list),
            {"source": [tf.Dimension(None)], "source_length": []},
            {"source": params.pad, "source_length": 0}
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        emo_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["emotion"]),
            default_value=10
        )
        pos_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["position"]),
            default_value=10
        )
        if data_type == "x":
            features["source"] = src_table.lookup(features["source"])
        if data_type == "ctx":
            features["source"] = src_table.lookup(features["source"])
        if data_type == "pos":
            features["source"] = pos_table.lookup(features["source"])
        if data_type == "emo":
            features["source"] = emo_table.lookup(features["source"])

        return features


def get_relevance_input(inputs, outputs, params):
    # inputs
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.constant(inputs)
    )

    # Split string
    dataset = dataset.map(lambda x: tf.string_split([x]).values,
                          num_parallel_calls=params.num_threads)

    # Append <eos>
    dataset = dataset.map(
        lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
        num_parallel_calls=params.num_threads
    )

    # Convert tuple to dictionary
    dataset = dataset.map(
        lambda x: {"source": x, "source_length": tf.shape(x)[0]},
        num_parallel_calls=params.num_threads
    )

    dataset = dataset.padded_batch(
        params.decode_batch_size,
        {"source": [tf.Dimension(None)], "source_length": []},
        {"source": params.pad, "source_length": 0}
    )

    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()

    src_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params.vocabulary["source"]),
        default_value=params.mapping["source"][params.unk]
    )
    features["source"] = src_table.lookup(features["source"])

    # outputs
    dataset_o = tf.data.Dataset.from_tensor_slices(
        tf.constant(outputs)
    )

    # Split string
    dataset_o = dataset_o.map(lambda x: tf.string_split([x]).values,
                          num_parallel_calls=params.num_threads)

    # Append <eos>
    dataset_o = dataset_o.map(
        lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
        num_parallel_calls=params.num_threads
    )

    # Convert tuple to dictionary
    dataset_o = dataset_o.map(
        lambda x: {"target": x, "target_length": tf.shape(x)[0]},
        num_parallel_calls=params.num_threads
    )

    dataset_o = dataset_o.padded_batch(
        params.decode_batch_size,
        {"target": [tf.Dimension(None)], "target_length": []},
        {"target": params.pad, "target_length": 0}
    )

    iterator = dataset_o.make_one_shot_iterator()
    features_o = iterator.get_next()

    src_table = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(params.vocabulary["target"]),
        default_value=params.mapping["target"][params.unk]
    )
    features["target"] = src_table.lookup(features_o["target"])
    features["target_length"] = features_o["target_length"]

    return features
