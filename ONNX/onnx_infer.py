
import numpy as np
import onnx
import onnxruntime
from onnx import helper

def get_num(str):
    s = str.split(" ")
    res = []
    for ss in s:
        num = int(ss)
        res.append(num)
    res = np.array(res, dtype=np.int64)
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

            src_shape = get_num(src.split(":")[0])
            s0 = src_shape[0]
            s1 = src_shape[1]
            s2 = src_shape[2]

            src_data = get_num(src.split(":")[1]).reshape(s0,s1,s2)
            pos_data = get_num(pos.split(":")[1]).reshape(s0,s1,s2)
            sent_data = get_num(sent.split(":")[1]).reshape(s0,s1,s2)
            mask_data = get_num(mask.split(":")[1]).reshape(s0,s1,s2)
            padding_tesnor = np.zeros((s0,128-s1,s2),dtype=np.int64)
            src_data = np.concatenate((src_data, padding_tesnor), axis=1)
            pos_data = np.concatenate((pos_data, padding_tesnor), axis=1)
            sent_data = np.concatenate((sent_data, padding_tesnor), axis=1)
            mask_data = np.concatenate((mask_data, padding_tesnor), axis=1)
            mask_data = np.array(mask_data, dtype=np.float32)

            aside_shape = get_num(aside1.split(":")[0])
            s0 = aside_shape[0]
            s1 = aside_shape[1]
            s2 = aside_shape[2]

            aside1_data = get_num(aside1.split(":")[1]).reshape(s0,s1,s2)
            aside2_data = get_num(aside2.split(":")[1]).reshape(s0,s1,s2)
            aside3_data = get_num(aside3.split(":")[1]).reshape(s0,s1,s2)
            aside4_data = get_num(aside4.split(":")[1]).reshape(s0,s1,s2)
            aside5_data = get_num(aside5.split(":")[1]).reshape(s0,s1,s2)
            aside6_data = get_num(aside6.split(":")[1]).reshape(s0,s1,s2)
            aside7_data = get_num(aside7.split(":")[1]).reshape(s0,s1,s2)
            aside8_data = get_num(aside8.split(":")[1]).reshape(s0,s1,s2)

            # print(src_data.shape, src_data.dtype)
            # print(pos_data.shape, pos_data.dtype)
            # print(sent_data.shape, sent_data.dtype)
            # print(mask.shape, mask.dtype)
            # print(aside1_data.shape, aside1_data.dtype)
            # print(aside2_data.shape, aside2_data.dtype)
            # print(aside3_data.shape, aside3_data.dtype)
            # print(aside4_data.shape, aside4_data.dtype)
            # print(aside5_data.shape, aside5_data.dtype)
            # print(aside6_data.shape, aside6_data.dtype)
            # print(aside7_data.shape, aside7_data.dtype)
            # print(aside8_data.shape, aside8_data.dtype)
            # assert(0)
            res.append([src_data, pos_data, sent_data, mask_data, aside1_data, aside2_data, aside3_data, aside4_data, aside5_data, aside6_data, aside7_data, aside8_data])
    return res

# src = [1,12,13,1557,40574,40997,22553,2,1571,40574,1569,42562,1557,40997,22553,1886,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# pos = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# sent = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# mask = [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
# aside = [1]
# src = np.array(src).reshape(1,128,1)
# pos = np.array(pos).reshape(1,128,1)
# sent = np.array(sent).reshape(1,128,1)
# mask = np.array(mask,dtype=np.float32).reshape(1,128,1)
# aside = np.array(aside).reshape(1,1,1)
# print(src.shape, src.dtype)
# print(pos.shape, pos.dtype)
# print(sent.shape, sent.dtype)
# print(mask.shape, mask.dtype)
# print(aside.shape, aside.dtype)
# assert(0)

inputs = load_data("/home/ubuntu/sti2_data/data/perf.test.txt")

# model check && model info
onnx_model = onnx.load("/home/ubuntu/sti2_data/model/onnx_infer_model/model.onnx")
onnx.checker.check_model(onnx_model)
graph = onnx_model.graph
input_info = graph.input
input_num = len(input_info)
output_info = graph.output
output_num = len(output_info)

# old_nodes = graph.node
# old_outs = graph.output
# new_idx = 1864
# new_nodes = old_nodes[:new_idx] # 去掉多余节点
# # new_outs = old_outs[:2]
# del onnx_model.graph.node[:] # 删除当前onnx模型的所有node
# onnx_model.graph.node.extend(new_nodes) # extend新的节点

# del onnx_model.graph.output[:] # 删除当前onnx模型的所有node
# # onnx_model.graph.output.extend(new_outs) # extend新的节点

# # print(len(old_nodes))
# # for i in range(300, 400): # 1863 236
# #     print(i, old_nodes[i].output)
# # assert(0)

# # graph.output[0].name = 'p2o.Add.211'
# # graph.output[0].type.tensor_type.elem_type = 1
# # graph.output[0].type.tensor_type.shape.dim[0].dim_value = -1
# # graph.output[0].type.tensor_type.shape.dim[1].dim_value = 128

# # graph.output[1].name = 'fc6_landmark_part2'
# # graph.output[1].type.tensor_type.elem_type = 1
# # graph.output[1].type.tensor_type.shape.dim[0].dim_value = 1
# # graph.output[1].type.tensor_type.shape.dim[1].dim_value = 146

# # prob_info =  helper.make_tensor_value_info('p2o.helper.squeeze.0',onnx.TensorProto.FLOAT, [-1, 128, 768])
# prob_info =  helper.make_tensor_value_info('p2o.Add.211',onnx.TensorProto.FLOAT, [-1, 128, 768])
# # 将构建完成的中间节点插入到模型中
# onnx_model.graph.output.insert(0, prob_info)
# onnx.save(onnx_model, '/home/ubuntu/sti2_data/model/onnx_infer_model/model_modified.onnx')

# print("===============================input===========================")
# print(input_info)
print("===============================output==========================")
print(output_info)
print("input num:", input_num)
print("output num:", output_num)

# create inference session
session = onnxruntime.InferenceSession("/home/ubuntu/sti2_data/model/onnx_infer_model/model.onnx")

# infer and save onnx baseline
# with open('onnx_baseline.txt', 'wb') as f:
#     for i in range(len(inputs)):
#         print(i)
#         input_list = {"read_file_0.tmp_0":inputs[i][0],
#                       "read_file_0.tmp_1":inputs[i][1],
#                       "read_file_0.tmp_2":inputs[i][2],
#                       "read_file_0.tmp_3":inputs[i][3],
#                       "read_file_0.tmp_6":inputs[i][4],
#                       "read_file_0.tmp_7":inputs[i][5],
#                       "read_file_0.tmp_8":inputs[i][6],
#                       "read_file_0.tmp_9":inputs[i][7],
#                       "read_file_0.tmp_10":inputs[i][8],
#                       "read_file_0.tmp_11":inputs[i][9],
#                       "read_file_0.tmp_12":inputs[i][10],
#                       "read_file_0.tmp_13":inputs[i][11]}
#         result = session.run(None, input_list)
#         res = result[0].reshape(1, result[0].shape[0])
#         np.savetxt(f, res, fmt='%.8f')