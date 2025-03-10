import argparse
import ctypes
import json
import numpy as np
import os
import os.path
import re
import sys
import time

import paddle

paddle.enable_static()

# TensorRT
import tensorrt as trt
from calibrator import ErnieCalibrator as ErnieCalibrator
from trt_helper import *


"""
TensorRT Initialization
"""
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# TRT_LOGGER = trt.Logger(trt.Logger.INFO)

# handle = ctypes.CDLL("libtrtplugin++.so", mode=ctypes.RTLD_GLOBAL)
# if not handle:
    # raise RuntimeError("Could not load plugin library. Is `LayerNorm.so` on your LD_LIBRARY_PATH?")

handle = ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
if not handle:
    raise RuntimeError("Could not load plugin library. Is `libnvinfer_plugin.so` on your LD_LIBRARY_PATH?")

slice_output_shape = None

trt.init_libnvinfer_plugins(TRT_LOGGER, "")
plg_registry = trt.get_plugin_registry()

def custom_fc(config, network, input_tensor, out_dims, w):
    pf_out_dims = trt.PluginField("out_dims", np.array([out_dims], dtype=np.int32), trt.PluginFieldType.INT32)
    pf_W = trt.PluginField("w", w.numpy(), trt.PluginFieldType.FLOAT32)
    pf_type = trt.PluginField("type_id", np.array([1 if config.fp16 else 0], np.int32), trt.PluginFieldType.INT32)
    pfc = trt.PluginFieldCollection([pf_out_dims, pf_W, pf_type])
    fc_plugin = fc_plg_creator.create_plugin("fcplugin", pfc)
    plug_inputs = [input_tensor]
    out_dense = network.add_plugin_v2(plug_inputs, fc_plugin)
    return out_dense


def build_attention_layer(args, network_helper, prefix, config, weights_dict, x, mask):
    local_prefix = prefix + "multi_head_att_"

    hidden_size = 768

    q_w = weights_dict[local_prefix + "query_fc.w_0"]
    q_b = weights_dict[local_prefix + "query_fc.b_0"]

    k_w = weights_dict[local_prefix + "key_fc.w_0"]
    k_b = weights_dict[local_prefix + "key_fc.b_0"]
    # k = network_helper.addLinear(x, k_w, k_b)

    v_w = weights_dict[local_prefix + "value_fc.w_0"]
    v_b = weights_dict[local_prefix + "value_fc.b_0"]
    # v = network_helper.addLinear(x, v_w, v_b)

    wqkv = np.zeros((3, hidden_size, hidden_size), np.float32)
    bqkv = np.zeros((3, hidden_size), np.float32)

    wqkv[0, :, :] = q_w.transpose(1, 0)
    wqkv[1, :, :] = k_w.transpose(1, 0)
    wqkv[2, :, :] = v_w.transpose(1, 0)
    bqkv[0, :] = q_b
    bqkv[1, :] = k_b
    bqkv[2, :] = v_b

    n = 12
    h = 64
    wall = np.ascontiguousarray(wqkv.reshape((3, n, h, n, h)).transpose((1, 0, 2, 3, 4)))
    ball = np.ascontiguousarray(bqkv.reshape((3, n, h)).transpose((1, 0, 2)))

    qkv_layer = network_helper.network.add_fully_connected(x, 3 * hidden_size, wall, ball)
    qkv = qkv_layer.get_output(0)

    qkv2ctx = network_helper.addQKV2CTX(args, qkv, mask)

    qkv2ctx = network_helper.addShuffle(qkv2ctx, None, (0, 0, 0), None, "pre_att_out_reshape")

    out_w = weights_dict[local_prefix + "output_fc.w_0"]
    out_b = weights_dict[local_prefix + "output_fc.b_0"]

    attn_output = network_helper.addLinear(qkv2ctx, out_w, out_b)

    attn_output = network_helper.addShuffle(attn_output, None, (0, 0, 0, 1, 1), None, "after_att_out_reshape")

    # attn (b, s, 1, 768)
    return attn_output


def build_mlp_layer(network_helper, prefix, config, weights_dict, x):

    local_prefix = prefix + "ffn_"
    # fc1_w = weights_dict[local_prefix + "fc_0.w_0"].transpose(1, 0)
    fc1_w = weights_dict[local_prefix + "fc_0.w_0"]
    fc1_b = weights_dict[local_prefix + "fc_0.b_0"]
    x = network_helper.addLinear(x, fc1_w, fc1_b)
    # x_layer = network_helper.network.add_fully_connected(x, 768*4, fc1_w, fc1_b)
    # x = x_layer.get_output(0)

    x = network_helper.addReLU(x)

    # fc2_w = weights_dict[local_prefix + "fc_1.w_0"].transpose(1, 0)
    fc2_w = weights_dict[local_prefix + "fc_1.w_0"]
    fc2_b = weights_dict[local_prefix + "fc_1.b_0"]
    x = network_helper.addLinear(x, fc2_w, fc2_b)
    # x_layer = network_helper.network.add_fully_connected(x, 768, fc2_w, fc2_b)
    # x = x_layer.get_output(0)

    return x


def build_embeddings_layer(network_helper, weights_dict, src_ids_tensor, sent_ids_tensor, pos_ids_tensor):
    #  weight info
    #  transformer.embeddings.position_embeddings [1, 197, 768]
    #  transformer.embeddings.cls_token [1, 1, 768]
    #  transformer.embeddings.patch_embeddings.weight [768, 3, 16, 16]
    #  transformer.embeddings.patch_embeddings.bias [768]

    word_embedding = weights_dict["word_embedding"]
    sent_embedding = weights_dict["sent_embedding"]
    pos_embedding = weights_dict["pos_embedding"]

    src_embedded = network_helper.addEmbedding(src_ids_tensor, word_embedding, "word_embedding")
    pos_embedded = network_helper.addEmbedding(pos_ids_tensor, pos_embedding, "pos_embedding")
    sent_embedded = network_helper.addEmbedding(sent_ids_tensor, sent_embedding, "sent_embedding")

    x = network_helper.addAdd(src_embedded, pos_embedded)
    x = network_helper.addAdd(x, sent_embedded)

    return x


def emb_layernorm(args, network_helper, weights_dict, src_ids_tensor, sent_ids_tensor, input_mask_tensor):
    src_ids_tensor = network_helper.addShuffle(src_ids_tensor, (1, 0, 2), (0, 0), None, "src_transpose")
    sent_ids_tensor = network_helper.addShuffle(sent_ids_tensor, (1, 0, 2), (0, 0), None, "sent_transpose")
    input_mask_tensor = network_helper.addShuffle(input_mask_tensor, (1, 0, 2), (0, 0), None, "mask_transpose")
    # network_helper.markOutput(input_mask_tensor)

    word_embedding = weights_dict["word_embedding"]
    sent_embedding = weights_dict["sent_embedding"]
    pos_embedding = weights_dict["pos_embedding"]
    pre_encoder_norm_weight = weights_dict["pre_encoder_layer_norm_scale"]
    pre_encoder_norm_bias = weights_dict["pre_encoder_layer_norm_bias"]

    wbeta = trt.PluginField("bert_embeddings_layernorm_beta", pre_encoder_norm_bias, trt.PluginFieldType.FLOAT32)
    wgamma = trt.PluginField("bert_embeddings_layernorm_gamma", pre_encoder_norm_weight, trt.PluginFieldType.FLOAT32)
    wwordemb = trt.PluginField("bert_embeddings_word_embeddings", word_embedding, trt.PluginFieldType.FLOAT32)
    wtokemb = trt.PluginField("bert_embeddings_token_type_embeddings", sent_embedding, trt.PluginFieldType.FLOAT32)
    wposemb = trt.PluginField("bert_embeddings_position_embeddings", pos_embedding, trt.PluginFieldType.FLOAT32)

    emb_layer = network_helper.addEmbeddingLayerNorm(args, src_ids_tensor, sent_ids_tensor, input_mask_tensor,
                                             wwordemb, wtokemb, wposemb, wgamma, wbeta, "embedding_layernorm")

    return emb_layer


def build_block_layer(args, network_helper, prefix, config, weights_dict, x, mask):
    local_prefix = prefix

    h = x

    # self.attn
    x = build_attention_layer(args, network_helper, local_prefix, config, weights_dict, x, mask)

    # x = network_helper.addAdd(x, h)

    # post_att_norm
    post_att_norm_weight = weights_dict[local_prefix + "post_att_layer_norm_scale"]
    post_att_norm_bias = weights_dict[local_prefix + "post_att_layer_norm_bias"]
    # x = network_helper.addLayerNorm(x, post_att_norm_weight, post_att_norm_bias)

    x = network_helper.addskipln(args, x, h, post_att_norm_weight, post_att_norm_bias)

    # _x = network_helper.addShuffle(x, None, (0, 0, 1, 768), None, "after_qkv2ctx_reshape")
    # network_helper.markOutput(_x)

    h = x

    # fnn
    x = network_helper.addShuffle(x, None, (0, 0, 0), None, "pre_mlp_reshape")
    x = build_mlp_layer(network_helper, local_prefix, config, weights_dict, x)
    x = network_helper.addShuffle(x, None, (0, 0, 0, 1, 1), None, "after_mlp_reshape")

    # x = network_helper.addAdd(x, h)

    # post ffn_norm
    fnn_norm_weight = weights_dict[local_prefix + "post_ffn_layer_norm_scale"]
    fnn_norm_bias = weights_dict[local_prefix + "post_ffn_layer_norm_bias"]
    x = network_helper.addskipln(args, x, h, fnn_norm_weight, fnn_norm_bias)

    return x


def build_encoder_layer(args, network_helper, prefix, config, weights_dict, x, mask):
    for layer in range(0, 12):
        local_prefix = prefix + "layer_{}_".format(layer)
        print("======================start===================")
        x = build_block_layer(args, network_helper, local_prefix, config, weights_dict, x, mask)
        print("======================end===================")
        # break

    # x = network_helper.addShuffle(x, (1, 0, 2, 3, 4), (0, 0, 1, 768), None, "block out trans_reshape")
    x = network_helper.addShuffle(x, None, (0, 0, 768), None, "block out trans_reshape")

    x_shape_len = len(x.shape)
    start = np.zeros(x_shape_len, dtype=np.int32)
    #  start_weight = trt.Weights(start)
    start_tensor = network_helper.addConstant(start)

    slice_layer = network_helper.network.add_slice(x, start, start, (1, 1, 1))
    slice_layer.set_input(1, start_tensor)
    slice_layer.set_input(2, slice_output_shape)
    sliced = slice_layer.get_output(0)

    pooled_w = weights_dict["pooled_fc.w_0"]
    pooled_b = weights_dict["pooled_fc.b_0"]
    x = network_helper.addLinear(sliced, pooled_w, pooled_b)

    return x


def build_ernie_model(args, network_helper, config, weights_dict, src_ids_tensor, sent_ids_tensor,
    pos_ids_tensor, input_mask_tensor):
    #  def forward(self, input_ids):
        #  embedding_output = self.embeddings(input_ids)
        #  encoded, attn_weights = self.encoder(embedding_output)
        #  return encoded, attn_weights
    prefix = "encoder_"

    embedding_layernorm = emb_layernorm(args, network_helper, weights_dict, src_ids_tensor,
        sent_ids_tensor, input_mask_tensor)
    embedding_out = embedding_layernorm.get_output(0)  # (-1, -1, 768, 1, 1)
    # embedding_out = network_helper.addShuffle(embedding_out, None, (0, 0, 1, 768),
    # None, "embedding_out_transpose")
    mask = embedding_layernorm.get_output(1)

    # trans_embedding_out = network_helper.addShuffle(embedding_out,
    # (1, 0, 2, 3), None, None, "embedding_out_transpose")
    # network_helper.markOutput(trans_embedding_out)
    # mask = network_helper.addDumpTensor(mask)
    # network_helper.markOutput(mask)
    # x:(-1, -1, 1, 768)
    encoder_out = build_encoder_layer(args, network_helper, prefix, config, weights_dict, embedding_out, mask)

    x = network_helper.addTanh(encoder_out)

    return x


def build_aside(network_helper, weights_dict, tensor_list):
    # pre_encoder_layer_norm_bias (768,)
    # feature_emb_fc_b2 (384,)
    # multi_field_1 (11, 20)
    # multi_field_2 (11, 20)
    # feature_emb_fc_w2 (768, 384)

    # multi_field_7 (11, 20)
    # cls_out_b_aside (1,)

    # pre_encoder_layer_norm_scale (768,)
    # feature_emb_fc_w (160, 768)
    # multi_field_5 (11, 20)
    # multi_field_0 (1432, 20)
    # cls_out_w_aside (384, 1)
    # multi_field_4 (11, 20)
    # multi_field_6 (11, 20)
    # multi_field_3 (13, 20)
    # feature_emb_fc_b (768,)

    multi_field_0 = weights_dict["multi_field_0"]
    multi_field_1 = weights_dict["multi_field_1"]
    multi_field_2 = weights_dict["multi_field_2"]
    multi_field_3 = weights_dict["multi_field_3"]
    multi_field_4 = weights_dict["multi_field_4"]
    multi_field_5 = weights_dict["multi_field_5"]
    multi_field_6 = weights_dict["multi_field_6"]
    multi_field_7 = weights_dict["multi_field_7"]

    if False:

        multi_field_0_tensor = network_helper.addConstant(multi_field_0)
        multi_field_1_tensor = network_helper.addConstant(multi_field_1)
        multi_field_2_tensor = network_helper.addConstant(multi_field_2)
        multi_field_3_tensor = network_helper.addConstant(multi_field_3)
        multi_field_4_tensor = network_helper.addConstant(multi_field_4)
        multi_field_5_tensor = network_helper.addConstant(multi_field_5)
        multi_field_6_tensor = network_helper.addConstant(multi_field_6)
        multi_field_7_tensor = network_helper.addConstant(multi_field_7)

        multi_fields = [multi_field_0_tensor, multi_field_1_tensor, multi_field_2_tensor, multi_field_3_tensor,
                        multi_field_4_tensor, multi_field_5_tensor, multi_field_6_tensor, multi_field_7_tensor]
        x = network_helper.addGatherCat(tensor_list, multi_fields)
        x = x.get_output(0)
    else:
        x0 = network_helper.addEmbedding(tensor_list[0], multi_field_0, "multi_field_0")
        x1 = network_helper.addEmbedding(tensor_list[1], multi_field_1, "multi_field_1")
        x2 = network_helper.addEmbedding(tensor_list[2], multi_field_2, "multi_field_2")
        x3 = network_helper.addEmbedding(tensor_list[3], multi_field_3, "multi_field_3")
        x4 = network_helper.addEmbedding(tensor_list[4], multi_field_4, "multi_field_4")
        x5 = network_helper.addEmbedding(tensor_list[5], multi_field_5, "multi_field_5")
        x6 = network_helper.addEmbedding(tensor_list[6], multi_field_6, "multi_field_6")
        x7 = network_helper.addEmbedding(tensor_list[7], multi_field_7, "multi_field_7")

        concat_tensors = [x0, x1, x2, x3, x4, x5, x6, x7]
        x = network_helper.addCat(concat_tensors, dim=1)

    # x = network_helper.addShuffle(x, None, (-1, 1, 1, 160), None, "aside_reshape")
    x = network_helper.addShuffle(x, None, (1, -1, 160), None, "aside_reshape")

    feature_emb_fc_w = weights_dict["feature_emb_fc_w"]
    feature_emb_fc_b = weights_dict["feature_emb_fc_b"]
    x = network_helper.addLinear(x, feature_emb_fc_w, feature_emb_fc_b)

    x = network_helper.addReLU(x)
    # x = network_helper.addShuffle(x, (1, 0, 2, 3), (1, -1, 768), None, "relu_transpose")
    # (1,-1,768)

    # get output shape, used in another slice
    global slice_output_shape
    slice_output_shape = network_helper.network.add_shape(x).get_output(0)

    feature_emb_fc_w2 = weights_dict["feature_emb_fc_w2"]
    feature_emb_fc_b2 = weights_dict["feature_emb_fc_b2"]
    x = network_helper.addLinear(x, feature_emb_fc_w2, feature_emb_fc_b2)

    x = network_helper.addReLU(x)

    cls_out_w_aside = weights_dict["cls_out_w_aside"]
    cls_out_b_aside = weights_dict["cls_out_b_aside"]
    x = network_helper.addLinear(x, cls_out_w_aside, cls_out_b_aside)
    return x


def build_model(args, network_helper, config, weights_dict, src_ids_tensor, sent_ids_tensor,
    pos_ids_tensor, input_mask_tensor, aside_tensor_list):
    #  def forward(self, x, labels=None):
        #  x, attn_weights = self.transformer(x)
        #  logits = self.head(x[:, 0])
    cls_aside_out = build_aside(network_helper, weights_dict, aside_tensor_list)
    # (1,b,1)

    x = build_ernie_model(args, network_helper, config, weights_dict, src_ids_tensor, sent_ids_tensor,
        pos_ids_tensor, input_mask_tensor)

    #  head.weight [10, 768]
    #  head.bias [10]
    cls_out_w = weights_dict["cls_out_w"]
    cls_out_b = weights_dict["cls_out_b"]
    cls_out = network_helper.addLinear(x, cls_out_w, cls_out_b)

    x = network_helper.addAdd(cls_out, cls_aside_out)

    x = network_helper.addSigmoid(x)

    return x


def build_engine(args, config, weights_dict, calibration_cache_file):
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch_flag) as network, \
    builder.create_builder_config() as builder_config:
        builder_config.max_workspace_size = args.workspace_size * (1024 * 1024)

        plugin_data_type: int = 0
        if args.fp16:
            plugin_data_type = 1
            builder_config.set_flag(trt.BuilderFlag.FP16)

        if args.int8:
            builder_config.set_flag(trt.BuilderFlag.INT8)

            # calibrator = ErnieCalibrator(args, calibration_cache_file, args.max_batch_size, 100)
            calibrator = ErnieCalibrator(args.calib_path, "calib_cache_file", 5, 128, 200)
            builder_config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
            builder_config.int8_calibrator = calibrator

        if args.strict:
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        #  if args.use_strict:
            #  builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        network_helper = TrtNetworkHelper(network, plg_registry, TRT_LOGGER, plugin_data_type)

        # Create the network
        src_ids_tensor = network_helper.addInput(name="src_ids", dtype=trt.int32, shape=(-1, -1, 1))
        # pos_ids_tensor = network_helper.addInput(name="pos_ids", dtype=trt.int32, shape=(-1, -1, 1))
        pos_ids_tensor = None
        sent_ids_tensor = network_helper.addInput(name="sent_ids", dtype=trt.int32, shape=(-1, -1, 1))
        # input_mask_tensor = network_helper.addInput(name="input_mask", dtype=trt.float32, shape=(-1, -1, 1))
        input_mask_tensor = network_helper.addInput(name="input_mask", dtype=trt.int32, shape=(-1, -1, 1))
        # input_mask_tensor2 = network_helper.addInput(name="input_mask2", dtype=trt.float32, shape=(-1, 1))

        tmp6_tensor = network_helper.addInput(name="tmp6", dtype=trt.int32, shape=(-1, 1, 1))
        tmp7_tensor = network_helper.addInput(name="tmp7", dtype=trt.int32, shape=(-1, 1, 1))
        tmp8_tensor = network_helper.addInput(name="tmp8", dtype=trt.int32, shape=(-1, 1, 1))
        tmp9_tensor = network_helper.addInput(name="tmp9", dtype=trt.int32, shape=(-1, 1, 1))
        tmp10_tensor = network_helper.addInput(name="tmp10", dtype=trt.int32, shape=(-1, 1, 1))
        tmp11_tensor = network_helper.addInput(name="tmp11", dtype=trt.int32, shape=(-1, 1, 1))
        tmp12_tensor = network_helper.addInput(name="tmp12", dtype=trt.int32, shape=(-1, 1, 1))
        tmp13_tensor = network_helper.addInput(name="tmp13", dtype=trt.int32, shape=(-1, 1, 1))

        aside_tensor_list = [tmp6_tensor, tmp7_tensor, tmp8_tensor, tmp9_tensor, tmp10_tensor,
            tmp11_tensor, tmp12_tensor, tmp13_tensor]

        out = build_model(args, network_helper, config, weights_dict, src_ids_tensor, sent_ids_tensor,
            pos_ids_tensor, input_mask_tensor, aside_tensor_list)

        network_helper.markOutput(out)

        if args.multi_context:
            # batchs = [[1, 5], [6, 10]]
            # seq_lens = [[1,  64], [32, 96], [64, 128]]
            # seq_lens = [[1, 128]]
            batchs = [[1, 10]]
            seq_lens = [[1, 63], [64, 64], [96, 96], [128, 128]]

            for b in batchs:
                for s in seq_lens:
                    profile = builder.create_optimization_profile()
                    min_shape = (b[0], s[0], 1)
                    opt_shape = (b[1], s[1], 1)
                    max_shape = (b[1], s[1], 1)
                    profile.set_shape("src_ids", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("sent_ids", min=min_shape, opt=opt_shape, max=max_shape)
                    # profile.set_shape("pos_ids", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("input_mask", min=min_shape, opt=opt_shape, max=max_shape)

                    min_shape = (b[0], 1, 1)
                    opt_shape = (b[1], 1, 1)
                    max_shape = (b[1], 1, 1)
                    profile.set_shape("tmp6", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp7", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp8", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp9", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp10", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp11", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp12", min=min_shape, opt=opt_shape, max=max_shape)
                    profile.set_shape("tmp13", min=min_shape, opt=opt_shape, max=max_shape)
                    builder_config.add_optimization_profile(profile)

        else:
            profile = builder.create_optimization_profile()
            min_shape = (1, 1, 1)
            # opt_shape = (5, 64, 1)
            opt_shape = (10, 96, 1)
            max_shape = (10, 128, 1)
            profile.set_shape("src_ids", min=min_shape, opt=opt_shape, max=max_shape)
            profile.set_shape("sent_ids", min=min_shape, opt=opt_shape, max=max_shape)
            # profile.set_shape("pos_ids", min=min_shape, opt=opt_shape, max=max_shape)
            profile.set_shape("input_mask", min=min_shape, opt=opt_shape, max=max_shape)

            min_shape = (1, 1, 1)
            # opt_shape = (5, 1, 1)
            opt_shape = (10, 1, 1)
            max_shape = (10, 1, 1)
            profile.set_shape("tmp6", min=min_shape, opt=opt_shape, max=max_shape)
            profile.set_shape("tmp7", min=min_shape, opt=opt_shape, max=max_shape)
            profile.set_shape("tmp8", min=min_shape, opt=opt_shape, max=max_shape)
            profile.set_shape("tmp9", min=min_shape, opt=opt_shape, max=max_shape)
            profile.set_shape("tmp10", min=min_shape, opt=opt_shape, max=max_shape)
            profile.set_shape("tmp11", min=min_shape, opt=opt_shape, max=max_shape)
            profile.set_shape("tmp12", min=min_shape, opt=opt_shape, max=max_shape)
            profile.set_shape("tmp13", min=min_shape, opt=opt_shape, max=max_shape)
            builder_config.add_optimization_profile(profile)

        build_start_time = time.time()

        #  import pdb
        #  pdb.set_trace()
        engine = builder.build_engine(network, builder_config)
        build_time_elapsed = (time.time() - build_start_time)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "build engine in {:.3f} Sec".format(build_time_elapsed))

        if args.int8:
            calibrator.free()
        return engine


def load_paddle_weights(path_prefix):
    """
    Load the weights from the onnx checkpoint
    """

    exe = paddle.static.Executor(paddle.CPUPlace())
    # path_prefix = "./paddle_infer_model"

    [inference_program, feed_target_names, _] = (paddle.static.load_inference_model(path_prefix, exe,
        model_filename="__model__", params_filename="__params__"))

    # import pdb
    # pdb.set_trace()

    state_dict = inference_program.state_dict()

    print(feed_target_names)

    tensor_dict = {}
    for i in state_dict:
        print(i)
        arr = np.array(state_dict[i])
        print(arr.shape)

        tensor_dict[i] = arr

    TRT_LOGGER.log(TRT_LOGGER.INFO, "Found {:} entries in weight map".format(len(tensor_dict)))

    return tensor_dict


def generate_calibration_cache(sequence_lengths, workspace_size, config, weights_dict, squad_json,
    vocab_file, calibration_cache_file, calib_num):
    """
    BERT demo needs a separate engine building path to generate calibration cache.
    This is because we need to configure SLN and MHA plugins in FP32 mode when
    generating calibration cache, and INT8 mode when building the actual engine.
    This cache could be generated by examining certain training data and can be
    reused across different configurations.
    """
    # dynamic shape not working with calibration, so we need generate a calibration
    # cache first using fulldims network
    if not config.int8 or os.path.exists(calibration_cache_file):
        return calibration_cache_file

    # generate calibration cache
    saved_use_fp16 = config.fp16
    config.fp16 = False
    config.is_calib_mode = True

    with build_engine([1], workspace_size, sequence_lengths, config, weights_dict, squad_json, vocab_file,
        calibration_cache_file, calib_num) as engine:
        TRT_LOGGER.log(TRT_LOGGER.INFO, "calibration cache generated in {:}".format(calibration_cache_file))

    config.fp16 = saved_use_fp16
    config.is_calib_mode = False


def main():
    parser = argparse.ArgumentParser(description="TensorRT BERT Sample",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--paddle", required=True, help="The paddle model dir path.")
    parser.add_argument("-o", "--output", required=True, default="bert_base_384.engine",
        help="The bert engine file, ex bert.engine")
    parser.add_argument("-b", "--max_batch_size", default=1, type=int, help="max batch size")
    parser.add_argument("-f", "--fp16", action="store_true",
        help="Indicates that inference should be run in FP16 precision", required=False)
    parser.add_argument("-i", "--int8", action="store_true",
        help="Indicates that inference should be run in INT8 precision", required=False)
    parser.add_argument("-t", "--strict", action="store_true",
        help="Indicates that inference should be run in strict precision mode", required=False)
    parser.add_argument("-w", "--workspace-size", default=12000,
        help="Workspace size in MiB for building the BERT engine", type=int)
    parser.add_argument("-c", "--calib_path", help="calibration cache path", required=False)
    parser.add_argument("-m", "--multi_context", action="store_true", help="calibration cache path", required=False)
    # parser.add_argument("-n", "--calib-num", help="calibration cache path", required=False)

    args, _ = parser.parse_known_args()

    # calib_cache = "ViT_N{}L{}A{}CalibCache".format(args.model_type,
    # config.transformer.num_layers, config.transformer.num_heads)
    # print(f"calib_cache = {calib_cache}")

    if args.paddle is not None:
        weights_dict = load_paddle_weights(args.paddle)
    else:
        raise RuntimeError("You need either specify paddle using option --paddle to build TRT model.")

    # config.encoder_num_layers = 12
    # config.num_heads = 12
    config = []
    with build_engine(args, config, weights_dict, None) as engine:
        TRT_LOGGER.log(TRT_LOGGER.VERBOSE, "Serializing Engine...")
        serialized_engine = engine.serialize()
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Saving Engine to {:}".format(args.output))
        with open(args.output, "wb") as fout:
            fout.write(serialized_engine)
        TRT_LOGGER.log(TRT_LOGGER.INFO, "Done.")

    # if args.img_path is not None:
    #     infer_helper = InferHelper(args.output, TRT_LOGGER)
    #     test_case_data(infer_helper, args, args.img_path)


if __name__ == "__main__":
    main()
