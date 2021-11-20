#!/usr/bin/env python
# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os,code
import six

import numpy as np
import tensorflow as tf
import thumt.data.dataset as dataset
import thumt.data.record as record
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.distribute as distribute
import thumt.utils.hooks as hooks
import thumt.utils.inference as inference
import thumt.utils.optimizers as optimizers
import thumt.utils.parallel as parallel


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Training neural machine translation models",
        usage="trainer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, nargs=2,
                        help="Path of source and target corpus")
    parser.add_argument("--record", type=str,
                        help="Path to tf.Record data")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to saved models")
    parser.add_argument("--dialog_src_context", type=str,
                        help="Path of context corpus")
    parser.add_argument("--dialog_tgt_context", type=str,
                        help="Path of context corpus")
    parser.add_argument("--style_src_context", type=str,
                        help="Path of context corpus")
    parser.add_argument("--style_tgt_context", type=str,
                        help="Path of context corpus")
    parser.add_argument("--language_src_context", type=str,
                        help="Path of context corpus")
    parser.add_argument("--language_tgt_context", type=str,
                        help="Path of context corpus")

    parser.add_argument("--dev_dialog_src_context", type=str,
                        help="Path of dev_context corpus")
    parser.add_argument("--dev_dialog_tgt_context", type=str,
                        help="Path of dev_context corpus")
    parser.add_argument("--dev_style_src_context", type=str,
                        help="Path of dev_context corpus")
    parser.add_argument("--dev_style_tgt_context", type=str,
                        help="Path of dev_context corpus")
    parser.add_argument("--dev_language_src_context", type=str,
                        help="Path of dev_context corpus")
    parser.add_argument("--dev_language_tgt_context", type=str,
                        help="Path of dev_context corpus")

    parser.add_argument("--emotion", type=str,
                        help="Path to emotion file")
    parser.add_argument("--dev_emotion", type=str,
                        help="Path to dev_emotion file")
    parser.add_argument("--vocabulary", type=str, nargs=2,
                        help="Path of source and target vocabulary")
    parser.add_argument("--validation", type=str,
                        help="Path of validation file")
    parser.add_argument("--references", type=str, nargs="+",
                        help="Path of reference files")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint")
    parser.add_argument("--half", action="store_true",
                        help="Enable FP16 training")
    parser.add_argument("--distribute", action="store_true",
                        help="Enable distributed training")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")

    return parser.parse_args(args)


def default_parameters():
    params = tf.contrib.training.HParams(
        input=["", ""],
        output="",
        record="",
        model="transformer",
        vocab=["", ""],
        # Default training hyper parameters
        num_threads=6,
        batch_size=4096,
        max_length=256,
        length_multiplier=1,
        mantissa_bits=2,
        start_steps=0,
        kl_annealing_steps=5000,
        warmup_steps=4000,
        train_steps=200000,
        buffer_size=10000,
        constant_batch_size=False,
        device_list=[0],
        update_cycle=1,
        initializer="uniform_unit_scaling",
        embedding_path="", # path to emotion embedding file
        initializer_gain=1.0,
        loss_scale=128,
        scale_l1=0.0,
        scale_l2=0.0,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        clip_grad_norm=5.0,
        learning_rate=1.0,
        learning_rate_decay="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        keep_checkpoint_max=1,
        keep_top_checkpoint_max=3,
        # Validation
        eval_steps=2000,
        eval_secs=0,
        eval_batch_size=32,
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=50,
        validation="",
        references=[""],
        save_checkpoint_secs=0,
        save_checkpoint_steps=5000,
        # Setting this to True can save disk spaces, but cannot restore
        # training using the saved checkpoint
        only_save_trainable=False
    )

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(p_name) or not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(p_name) as fd:
        tf.logging.info("Restoring hyper parameters from %s" % p_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def export_params(output_dir, name, params):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)
    with tf.gfile.Open(filename, "w") as fd:
        fd.write(params.to_json())


def collect_params(all_params, params):
    collected = tf.contrib.training.HParams()

    for k in six.iterkeys(params.values()):
        collected.add_hparam(k, getattr(all_params, k))

    return collected


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def override_parameters(params, args):
    params.model = args.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.dialog_src_context = args.dialog_src_context # or params.dialog_src_context
    params.dialog_tgt_context = args.dialog_tgt_context #or params.dialog_tgt_context
    params.style_src_context = args.style_src_context #or params.style_src_context
    params.style_tgt_context = args.style_tgt_context #or params.style_tgt_context
    params.language_src_context = args.language_src_context #or params.language_src_context
    params.language_tgt_context = args.language_tgt_context #or params.language_tgt_context

    params.dev_dialog_src_context = args.dev_dialog_src_context #or params.dev_dialog_src_context
    params.dev_dialog_tgt_context = args.dev_dialog_tgt_context #or params.dev_dialog_tgt_context
    params.dev_style_src_context = args.dev_style_src_context #or params.dev_style_src_context
    params.dev_style_tgt_context = args.dev_style_tgt_context #or params.dev_style_tgt_context
    params.dev_language_src_context = args.dev_language_src_context #or params.dev_language_src_context
    params.dev_language_tgt_context = args.dev_language_tgt_context #or params.dev_language_tgt_context

    params.emotion = args.emotion or params.emotion
    params.dev_emotion = args.dev_emotion or params.dev_emotion
    params.record = args.record or params.record
    params.vocab = args.vocabulary or params.vocab
    params.validation = args.validation or params.validation
    params.references = args.references or params.references
    params.parse(args.parameters)

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(params.vocab[0]),
        "target": vocabulary.load_vocabulary(params.vocab[1]),
        "emotion": ['neutral', 'joy', 'anger', 'surprise', 'sadness', 'fear', 'disgust', 'happiness', 'happy', 'other'],
        "position": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    params.vocabulary["target"] = vocabulary.process_vocabulary(
        params.vocabulary["target"], params
    )
    params.vocabulary["emotion"] = vocabulary.process_vocabulary(
        params.vocabulary["emotion"], params
    )
    params.vocabulary["position"] = vocabulary.process_vocabulary(
        params.vocabulary["position"], params
    )
    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            control_symbols
        ),
        "target": vocabulary.get_control_mapping(
            params.vocabulary["target"],
            control_symbols
        )
    }

    return params


def get_initializer(params):
    if params.initializer == "uniform":
        max_val = params.initializer_gain
        return tf.random_uniform_initializer(-max_val, max_val)
    elif params.initializer == "normal":
        return tf.random_normal_initializer(0.0, params.initializer_gain)
    elif params.initializer == "normal_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="normal")
    elif params.initializer == "uniform_unit_scaling":
        return tf.variance_scaling_initializer(params.initializer_gain,
                                               mode="fan_avg",
                                               distribution="uniform")
    else:
        raise ValueError("Unrecognized initializer: %s" % params.initializer)


def get_learning_rate_decay(learning_rate, global_step, params):
    if params.learning_rate_decay in ["linear_warmup_rsqrt_decay", "noam"]:
        step = tf.to_float(global_step)
        warmup_steps = tf.to_float(params.warmup_steps)
        multiplier = params.hidden_size ** -0.5
        decay = multiplier * tf.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)

        return learning_rate * decay
    elif params.learning_rate_decay == "piecewise_constant":
        return tf.train.piecewise_constant(tf.to_int32(global_step),
                                           params.learning_rate_boundaries,
                                           params.learning_rate_values)
    elif params.learning_rate_decay == "none":
        return learning_rate
    else:
        raise ValueError("Unknown learning_rate_decay")


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=True)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)

    if distribute.is_distributed_training_mode():
        config.gpu_options.visible_device_list = str(distribute.local_rank())
    elif params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def decode_target_ids(inputs, params):
    decoded = []
    vocab = params.vocabulary["target"]

    for item in inputs:
        syms = []
        for idx in item:
            if isinstance(idx, six.integer_types):
                sym = vocab[idx]
            else:
                sym = idx
                if not isinstance(sym, six.string_types):
                    sym = sym.decode("utf-8")

            if sym == params.eos:
                break

            if sym == params.pad:
                break

            syms.append(sym)
        decoded.append(syms)

    return decoded


def restore_variables(checkpoint):
    if not checkpoint:
        return tf.no_op("restore_op")

    # Load checkpoints
    tf.logging.info("Loading %s" % checkpoint)
    var_list = tf.train.list_variables(checkpoint)
    reader = tf.train.load_checkpoint(checkpoint)
    values = {}

    for (name, shape) in var_list:
        tensor = reader.get_tensor(name)
        name = name.split(":")[0]
        values[name] = tensor

    var_list = tf.trainable_variables()
    ops = []

    for var in var_list:
        name = var.name.split(":")[0]

        if name in values:
            tf.logging.info("Restore %s" % var.name)
            ops.append(tf.assign(var, values[name]))

    return tf.group(*ops, name="restore_op")


def print_variables():
    all_weights = {v.name: v for v in tf.trainable_variables()}
    total_size = 0

    for v_name in sorted(list(all_weights)):
        v = all_weights[v_name]
        tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                        str(v.shape).ljust(20))
        v_size = np.prod(np.array(v.shape.as_list())).tolist()
        total_size += v_size
    tf.logging.info("Total trainable variables size: %d", total_size)

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
    dialog = []
    for src, ctx in zip(sentence, content):
        line = ctx.replace('\n', ' ') + src.replace('\n', ' ') + '<eos> '
        dialog.append(line)
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
    dialog_file = base_path + '/' + signal + '.src.dialog'

    with open(position_file, 'w', encoding='utf-8') as fw:
        for line_position in turn_position:
            line_position = sorted(line_position, reverse=True)
            fw.write(' '.join(line_position) + '\n')

    with open(dialog_file, 'w', encoding='utf-8') as fw:
        for sub_mask in dialog:
            fw.write(sub_mask)

    #code.interact(local=locals())
    return position_file, dialog_file

def get_turn_position1(file1):
    base_path = '/'.join(file1.split('/')[:-1])
    signal = file1.split('/')[-1].split('.')[0]
    position_file = base_path + '/' + signal + '.turn_position'
    return position_file

def main(args):
    if args.distribute:
        distribute.enable_distributed_training()

    tf.logging.set_verbosity(tf.logging.INFO)
    model_cls = models.get_model(args.model)
    params = default_parameters()

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = merge_parameters(params, model_cls.get_parameters())
    params = import_params(args.output, args.model, params)
    override_parameters(params, args)

    # Export all parameters and model specific parameters
    if distribute.rank() == 0:
        export_params(params.output, "params.json", params)
        export_params(
            params.output,
            "%s.json" % args.model,
            collect_params(params, model_cls.get_parameters())
        )

    # Build Graph
    with tf.Graph().as_default():
        if not params.record:
            # Build input queue
            #get_turn_position(params.context)
            #features = dataset.get_training_input(params.input, params)
            features = dataset.get_training_input_contextual(params.input, params.emotion, params)
            #files = params.input
            #train_inputs = dataset.sort_and_zip_files(files)
            #features = dataset.get_training_input_contextual_emo(train_inputs, params)
        else:
            features = record.get_input_features(
                os.path.join(params.record, "*train*"), "train", params
            )

        # Build model
        initializer = get_initializer(params)
        regularizer = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=params.scale_l1, scale_l2=params.scale_l2)
        model = model_cls(params)
        # Create global step
        global_step = tf.train.get_or_create_global_step()
        dtype = tf.float16 if args.half else None

        # Multi-GPU setting
        s_losses = parallel.parallel_model(
            model.get_training_func(initializer, regularizer, dtype),
            features,
            params.device_list
        )
        sharded_losses = []
        kl_losses = []
        bow_losses = []
        for item in s_losses:
            sharded_loss, kl_loss, bow_loss = item[0], item[1], item[2]
            sharded_losses.append(sharded_loss)
            kl_losses.append(kl_loss)
            bow_losses.append(bow_loss)

        ce_loss = tf.add_n(sharded_losses) / len(sharded_losses)
        kl_loss = tf.add_n(kl_losses) / len(kl_losses)
        bow_loss = tf.add_n(bow_losses) / len(bow_losses)

        ce_loss = ce_loss + tf.losses.get_regularization_loss()

        if distribute.rank() == 0:
            print_variables()

        learning_rate = get_learning_rate_decay(params.learning_rate,
                                                global_step, params)
        learning_rate = tf.convert_to_tensor(learning_rate, dtype=tf.float32)

        tf.summary.scalar("loss", ce_loss)
        tf.summary.scalar("learning_rate", learning_rate)

        # Create optimizer
        if params.optimizer == "Adam":
            opt = tf.train.AdamOptimizer(learning_rate,
                                         beta1=params.adam_beta1,
                                         beta2=params.adam_beta2,
                                         epsilon=params.adam_epsilon)
        elif params.optimizer == "LazyAdam":
            opt = tf.contrib.opt.LazyAdamOptimizer(learning_rate,
                                                   beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2,
                                                   epsilon=params.adam_epsilon)
        else:
            raise RuntimeError("Optimizer %s not supported" % params.optimizer)

        opt = optimizers.MultiStepOptimizer(opt, params.update_cycle)

        if args.half:
            opt = optimizers.LossScalingOptimizer(opt, params.loss_scale)
        kstep = global_step
        if params.start_steps > 199999:
              kstep = global_step - params.start_steps
        kl_weights = tf.minimum(tf.to_float(kstep) / params.kl_annealing_steps, 1.0)
        #total_loss = ce_loss + kl_loss * kl_weights + bow_loss * 0.5
        if params.use_bowloss:
            total_loss = ce_loss * 0.9 + kl_loss * kl_weights + bow_loss * 0.1
        else:
            total_loss = ce_loss + kl_loss * kl_weights

        # Optimization
        grads_and_vars = opt.compute_gradients(
            total_loss, colocate_gradients_with_ops=True)

        if params.clip_grad_norm:
            grads, var_list = list(zip(*grads_and_vars))
            grads, _ = tf.clip_by_global_norm(grads, params.clip_grad_norm)
            grads_and_vars = zip(grads, var_list)

        train_op = opt.apply_gradients(grads_and_vars,
                                       global_step=global_step)

        # Validation
        if params.validation and params.references[0]:
            #files = [params.validation] + list(params.references)
            position_file_src_dia = get_turn_position(params.dev_dialog_src_context)
#            position_file_src_dia, dialog_file = get_turn_position_src(params.validation, params.dialog_src_context)

            position_file_tgt_dia = get_turn_position(params.dev_dialog_tgt_context)
            position_file_src_sty = get_turn_position(params.dev_style_src_context)
            position_file_tgt_sty = get_turn_position(params.dev_style_tgt_context)
            position_file_src_lan = get_turn_position(params.dev_language_src_context)
            position_file_tgt_lan = get_turn_position(params.dev_language_tgt_context)

            files = [params.validation] + [params.dev_dialog_src_context] + [position_file_src_dia] + [params.dev_dialog_tgt_context] + [position_file_tgt_dia] + [params.dev_style_src_context] + [position_file_src_sty] + [params.dev_style_tgt_context] + [position_file_tgt_sty] + [params.dev_language_src_context] + [position_file_src_lan] + [params.dev_language_tgt_context] + [position_file_tgt_lan] + [params.dev_emotion] + list(params.references)
            eval_inputs = dataset.sort_and_zip_files(files)
            #code.interact(local=locals())
            #eval_input_fn = dataset.get_evaluation_input
            eval_input_fn = dataset.get_evaluation_input_ctx
        else:
            eval_input_fn = None

        # Hooks
        train_hooks = [
            tf.train.StopAtStepHook(last_step=params.train_steps),
            tf.train.NanTensorHook(total_loss),
            tf.train.LoggingTensorHook(
                {
                    "step": global_step,
                    "lr": learning_rate,
                    "ce_loss": ce_loss,
                    "kl_loss": kl_loss,
                    "kl_weights": kl_weights,
                    "bow_loss": bow_loss,
                    "total_loss": total_loss,
                    "source": tf.shape(features["source"]),
                    "dialog": tf.shape(features["context_dia_src"]),
                },
                every_n_iter=1
            )
        ]

        broadcast_hook = distribute.get_broadcast_hook()

        if broadcast_hook:
            train_hooks.append(broadcast_hook)

        if distribute.rank() == 0:
            # Add hooks
            save_vars = tf.trainable_variables() + [global_step]
            saver = tf.train.Saver(
                var_list=save_vars if params.only_save_trainable else None,
                max_to_keep=params.keep_checkpoint_max,
                sharded=False
            )
            tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
            train_hooks.append(
                hooks.MultiStepHook(
                    tf.train.CheckpointSaverHook(
                        checkpoint_dir=params.output,
                        save_secs=params.save_checkpoint_secs or None,
                        save_steps=params.save_checkpoint_steps or None,
                        saver=saver),
                step=params.update_cycle)
            )

            if eval_input_fn is not None:
                train_hooks.append(
                    hooks.MultiStepHook(
                        hooks.EvaluationHook(
                            lambda f: inference.create_inference_graph(
                                [model], f, params
                            ),
                            lambda: eval_input_fn(eval_inputs, params),
                            lambda x: decode_target_ids(x, params),
                            params.output,
                            session_config(params),
                            params.keep_top_checkpoint_max,
                            eval_secs=params.eval_secs,
                            eval_steps=params.eval_steps
                        ),
                        step=params.update_cycle
                    )
                )
            checkpoint_dir = params.output
        else:
            checkpoint_dir = None

        restore_op = restore_variables(args.checkpoint)

        def restore_fn(step_context):
            step_context.session.run(restore_op)

        # Create session, do not use default CheckpointSaverHook
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=checkpoint_dir, hooks=train_hooks,
                save_checkpoint_secs=None,
                config=session_config(params)) as sess:
            # Restore pre-trained variables
            sess.run_step_fn(restore_fn)

            while not sess.should_stop():
                sess.run(train_op)


if __name__ == "__main__":
    main(parse_args())
