import paddle
import torch
import numpy as np
from torch import nn
from torch.nn import CrossEntropyLoss
import math
import copy
from collections import namedtuple

import argparse

WeightDict = {}
def load_paddle_weights(path_prefix):
    """
    Load the weights from the onnx checkpoint
    """
    exe = paddle.static.Executor(paddle.CPUPlace())
    # path_prefix = "./sti2_data/model/paddle_infer_model"
    paddle.enable_static()
    [inference_program, feed_target_names, fetch_targets] = (
                paddle.static.load_inference_model(path_prefix, exe, model_filename="__model__", params_filename="__params__"))

    state_dict = inference_program.state_dict()
    #print(feed_target_names)
    global WeightDict

    for i in state_dict:
        # print(i)
        arr = np.array(state_dict[i])
        # print(arr.shape)
        WeightDict[i] = torch.tensor(arr)

    return WeightDict

def getConfig():
    Config = namedtuple("Config", ["transformer", "aside", "encoder_num_layers",])
    TransformerConfig = namedtuple("TransformerConfig", ["num_heads", "hidden_size", "head_size", "intermediate_size"])
    AsideConfig = namedtuple("AsideConfig", ["embed_fc", "cls_out_r", "cls_out_l"])
    transformer_config = TransformerConfig(num_heads=12,
                                           head_size=64,
                                           hidden_size=768,
                                            intermediate_size=3072,
                                           )
    aside_config = AsideConfig(embed_fc=160, cls_out_r=384, cls_out_l=1)
    config = Config(encoder_num_layers=12,
                    transformer=transformer_config,
                    aside=aside_config)
    return config
class LayerNormLayer(nn.Module):
    def __init__(self, config, norm_weight, bias_weight, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNormLayer, self).__init__()
        self.gamma = nn.Parameter(norm_weight)
        self.beta = nn.Parameter(bias_weight)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class AttentionLayer(nn.Module):
    def __init__(self, config, prefix):
        super(AttentionLayer, self).__init__()

        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = config.transformer.head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        local_prefix = prefix + "multi_head_att_"
        self.query = nn.Linear(config.transformer.hidden_size, config.transformer.hidden_size)
        self.query.weight.data = WeightDict[local_prefix + "query_fc.w_0"].t()
        self.query.bias.data = WeightDict[local_prefix + "query_fc.b_0"]

        self.key = nn.Linear(config.transformer.hidden_size, config.transformer.hidden_size)
        self.key.weight.data = WeightDict[local_prefix + "key_fc.w_0"].t()
        self.key.bias.data = WeightDict[local_prefix + "key_fc.b_0"]

        self.value = nn.Linear(config.transformer.hidden_size, config.transformer.hidden_size)
        self.value.weight.data = WeightDict[local_prefix + "value_fc.w_0"].t()
        self.value.bias.data = WeightDict[local_prefix + "value_fc.b_0"]

        self.dense = nn.Linear(config.transformer.hidden_size, config.transformer.hidden_size)
        self.dense.weight.data = WeightDict[local_prefix + "output_fc.w_0"].t()
        self.dense.bias.data = WeightDict[local_prefix + "output_fc.b_0"]

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention * V
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # [B,S,D] * [D,D]
        attn_output = self.dense(context_layer)
        return attn_output

class MLPLayer(nn.Module):
    def __init__(self, config, prefix):
        super(MLPLayer, self).__init__()
        local_prefix = prefix + "ffn_"
        self.fc1 = nn.Linear(config.transformer.intermediate_size, config.transformer.hidden_size)
        self.fc1.weight.data = WeightDict[local_prefix + "fc_0.w_0"].t()
        self.fc1.bias.data = WeightDict[local_prefix + "fc_0.b_0"]
        self.act = nn.ReLU()

        self.fc2 = nn.Linear(config.transformer.hidden_size, config.transformer.intermediate_size)
        self.fc2.weight.data = WeightDict[local_prefix + "fc_1.w_0"].t()
        self.fc2.bias.data = WeightDict[local_prefix + "fc_1.b_0"]
    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super(EmbeddingLayer, self).__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(WeightDict["word_embedding"])
        self.position_embeddings = nn.Embedding.from_pretrained(WeightDict["pos_embedding"])
        self.sent_embeddings = nn.Embedding.from_pretrained(WeightDict["sent_embedding"])
    def forward(self, input_ids, sent_ids_tensor, position_ids):
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        sent_embedded = self.sent_embeddings(sent_ids_tensor)

        embeddings = words_embeddings + position_embeddings + sent_embedded

        return embeddings

class BlockLayer(nn.Module):
    def __init__(self, config, prefix):
        super(BlockLayer, self).__init__()
        local_prefix = prefix
        post_att_norm_weight = WeightDict[local_prefix + "post_att_layer_norm_scale"]
        post_att_norm_bias = WeightDict[local_prefix + "post_att_layer_norm_bias"]
        self.post_att_layernorm = LayerNormLayer(config, post_att_norm_weight, post_att_norm_bias)

        post_att_norm_weight = WeightDict[local_prefix + "post_att_layer_norm_scale"]
        post_att_norm_bias = WeightDict[local_prefix + "post_att_layer_norm_bias"]
        self.post_ffn_layernorm = LayerNormLayer(config, post_att_norm_weight, post_att_norm_bias)
        self.atten = AttentionLayer(config, local_prefix)
        self.mlp = MLPLayer(config, local_prefix)
    def forward(self, x, mask):

        residual = self.atten(x, mask)
        x += residual
        x = self.post_att_layernorm(x)

        residual = self.mlp(x)
        x += residual
        x = self.post_ffn_layernorm(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, config, prefix):
        super(EncoderLayer, self).__init__()
        layer_list = []
        for idx in range(config.encoder_num_layers):
            local_prefix = prefix + "layer_{}_".format(idx)
            layer_list.append(BlockLayer(config, local_prefix))
        self.layer = nn.ModuleList(layer_list)
        self.pooled = nn.Linear(config.transformer.hidden_size, config.transformer.hidden_size)

        self.pooled.weight.data = WeightDict["pooled_fc.w_0"].t()
        self.pooled.bias.data = WeightDict["pooled_fc.b_0"]

    def forward(self, hidden_states, attention_mask, slice_output_shape):

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
        output_elements = 1
        for shape in slice_output_shape:
            output_elements *= shape
        sliced = hidden_states.view(-1)[:output_elements].reshape(slice_output_shape)
        output = self.pooled(sliced)
        return output

class ErnireModel(nn.Module):
    def __init__(self, config):
        super(ErnireModel, self).__init__()
        prefix = "encoder_"
        self.embedding = EmbeddingLayer(config)
        pre_encoder_norm_weight = WeightDict["pre_encoder_layer_norm_scale"]
        pre_encoder_norm_bias = WeightDict["pre_encoder_layer_norm_bias"]
        self.pre_encoder_ln = LayerNormLayer(config, pre_encoder_norm_weight, pre_encoder_norm_bias)
        self.encoder_layer = EncoderLayer(config, prefix)
        self.activation = nn.Tanh()

    def forward(self, src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor, slice_output_shape):
        embedding = self.embedding(src_ids_tensor, sent_ids_tensor, pos_ids_tensor)
        # if len(embedding.shape) == 4 and embedding.shape[2] == 1:
        #     embedding = embedding.squeeze(dim=2)
        x = self.pre_encoder_ln(embedding)
        encoder_out = self.encoder_layer(x, input_mask_tensor, slice_output_shape)
        x = self.activation(encoder_out)
        return x


class Aside(nn.Module):
    def __init__(self, config):
        super(Aside, self).__init__()
        self.embed0 = nn.Embedding.from_pretrained(WeightDict["multi_field_0"])
        self.embed1 = nn.Embedding.from_pretrained(WeightDict["multi_field_1"])
        self.embed2 = nn.Embedding.from_pretrained(WeightDict["multi_field_2"])
        self.embed3 = nn.Embedding.from_pretrained(WeightDict["multi_field_3"])
        self.embed4 = nn.Embedding.from_pretrained(WeightDict["multi_field_4"])
        self.embed5 = nn.Embedding.from_pretrained(WeightDict["multi_field_5"])
        self.embed6 = nn.Embedding.from_pretrained(WeightDict["multi_field_6"])
        self.embed7 = nn.Embedding.from_pretrained(WeightDict["multi_field_7"])

        self.feature_emb_fc1 = nn.Linear(config.transformer.hidden_size, config.aside.embed_fc)# 160 768 .t()
        self.feature_emb_fc1.weight.data = WeightDict["feature_emb_fc_w"].t()
        self.feature_emb_fc1.bias.data = WeightDict["feature_emb_fc_b"]

        self.activation = nn.ReLU()

        self.feature_emb_fc2 = nn.Linear(config.aside.cls_out_r, config.transformer.hidden_size)# 768 384 .t()
        self.feature_emb_fc2.weight.data = WeightDict["feature_emb_fc_w2"].t()
        self.feature_emb_fc2.bias.data = WeightDict["feature_emb_fc_b2"]

        self.cls_out = nn.Linear(config.aside.cls_out_l, config.aside.cls_out_r)# 384 1
        self.cls_out.weight.data = WeightDict["cls_out_w_aside"].t()
        self.cls_out.bias.data = WeightDict["cls_out_b_aside"]
    def forward(self, vector_list):
        x0 = self.embed0(vector_list[0])
        x1 = self.embed1(vector_list[1])
        x2 = self.embed2(vector_list[2])
        x3 = self.embed3(vector_list[3])
        x4 = self.embed4(vector_list[4])
        x5 = self.embed5(vector_list[5])
        x6 = self.embed0(vector_list[6])
        x7 = self.embed7(vector_list[7])
        x_cat = torch.cat((x0, x1, x2, x3, x4, x5, x6, x7), dim = 1)
        x = x_cat.reshape(-1, 1, 1, 160)

        x = self.feature_emb_fc1(x)
        x = self.activation(x)
        slice_output_shape = x.shape

        x = self.feature_emb_fc2(x)
        x = self.activation(x)

        x = self.cls_out(x)
        return x, slice_output_shape

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        config = getConfig()
        self.aside = Aside(config)
        self.ernie_model = ErnireModel(config)

        self.cls_out = nn.Linear(1, config.transformer.hidden_size)# 768 1
        self.cls_out.weight.data = WeightDict["cls_out_w"].t()
        self.cls_out.bias.data = WeightDict["cls_out_b"]
    def process_input_mask(self, input_mask_tensor):
        input_mask_tensor = input_mask_tensor.unsqueeze(-1)
        attn_bias = torch.matmul(input_mask_tensor, input_mask_tensor.transpose(1, 2))
        attn_bias = (1. - attn_bias) * -10000.0
        attn_bias = attn_bias.unsqueeze(1)
        return attn_bias

    def forward(self, src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor, aside_tensor_list):
        cls_aside_out, slice_output_shape = self.aside(aside_tensor_list)
        input_mask_tensor = self.process_input_mask(input_mask_tensor)
        x = self.ernie_model(src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor, slice_output_shape)

        cls_out = self.cls_out(x)
        x = cls_out + cls_aside_out
        x = torch.sigmoid(x)
        return x

def get_num(str, type):
    s = str.split(" ")
    res = []
    for ss in s:
        if type == "int":
            num = int(ss)
        else:
            num = float(ss)
        res.append(num)
    if type == "int":
        res = np.array(res, dtype=np.int32)
    else:
        res = np.array(res, dtype=np.float32)
    return res

def load_data(file_path):
    res = []
    with open(file_path,"r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(";")
            src = data[2]
            pos = data[3]
            sent = data[4]
            mask = data[5]
            aside1 = data[6]
            aside2 = data[7]
            aside3 = data[8]
            aside4 = data[9]
            aside5 = data[10]
            aside6 = data[11]
            aside7 = data[12]
            aside8 = data[13]

            src_shape = get_num(src.split(":")[0], "int")
            s0 = src_shape[0]
            s1 = src_shape[1]
            s2 = src_shape[2]

            src_data = get_num(src.split(":")[1], "int").reshape(s0,s1)
            pos_data = get_num(pos.split(":")[1], "int").reshape(s0,s1)
            sent_data = get_num(sent.split(":")[1], "int").reshape(s0,s1)
            mask_data = get_num(mask.split(":")[1], "float32").reshape(s0,s1)

            # # convert mask (-1, -1, 1) to (-1, 1)
            # mask = []
            # for i in range(s0):
                # sum = 0
                # for j in range(s1):
                    # for k in range(s2):
                        # sum += mask_data[i][j][k]
                # mask.append(sum)
            # mask = np.array(mask, dtype=np.int32).reshape(-1, 1)

            aside_shape = get_num(aside1.split(":")[0], "int")
            s0 = aside_shape[0]
            s1 = aside_shape[1]
            s2 = aside_shape[2]

            aside1_data = get_num(aside1.split(":")[1], "int").reshape(s0,s1)
            aside2_data = get_num(aside2.split(":")[1], "int").reshape(s0,s1)
            aside3_data = get_num(aside3.split(":")[1], "int").reshape(s0,s1)
            aside4_data = get_num(aside4.split(":")[1], "int").reshape(s0,s1)
            aside5_data = get_num(aside5.split(":")[1], "int").reshape(s0,s1)
            aside6_data = get_num(aside6.split(":")[1], "int").reshape(s0,s1)
            aside7_data = get_num(aside7.split(":")[1], "int").reshape(s0,s1)
            aside8_data = get_num(aside8.split(":")[1], "int").reshape(s0,s1)

            res.append([src_data, pos_data, sent_data, mask_data, aside1_data, aside2_data, aside3_data, aside4_data, aside5_data, aside6_data, aside7_data, aside8_data])
    return res

def main(args):

    load_paddle_weights(args.paddle_file)

    model = Model()

    input_datas = load_data(args.input_file)

    for data in input_datas:
        src_ids_tensor = torch.from_numpy(data[0])
        pos_ids_tensor = torch.from_numpy(data[1])
        sent_ids_tensor = torch.from_numpy(data[2])
        input_mask_tensor = torch.from_numpy(data[3])

        tmp6_tensor = torch.from_numpy(data[4])
        tmp7_tensor = torch.from_numpy(data[5])
        tmp8_tensor = torch.from_numpy(data[6])
        tmp9_tensor = torch.from_numpy(data[7])
        tmp10_tensor = torch.from_numpy(data[8])
        tmp11_tensor = torch.from_numpy(data[9])
        tmp12_tensor = torch.from_numpy(data[10])
        tmp13_tensor = torch.from_numpy(data[11])

        aside_tensor_list = [tmp6_tensor, tmp7_tensor, tmp8_tensor, tmp9_tensor, tmp10_tensor, tmp11_tensor, tmp12_tensor, tmp13_tensor]

        with torch.no_grad():
            output = model(src_ids_tensor, sent_ids_tensor, pos_ids_tensor, input_mask_tensor, aside_tensor_list)
            print(output)

        break


# test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ernie trt model test")

    parser.add_argument("-p", "--paddle_file", required=True, help="The paddle model file path.")
    parser.add_argument("-i", "--input_file", required=True, help="The onnx baseline file path.")

    args = parser.parse_args()
    main(args)
