import argparse
import numpy as np
import time

import tensorrt as trt
import trt_helper

logger = trt_helper.init_trt_plugin(trt.Logger.VERBOSE, "libtrtplugin++.so.1")
def cal_dif(onnx, trt, l):
    print("onnx_res:", onnx)
    print("trt_res: ", trt)
    dif_sum = 0
    min_dif = 1
    max_dif = 0
    for i in range(l):
        dif = abs(onnx[i] - trt[i]) / abs(onnx[i])
        max_dif = max(max_dif, dif)
        min_dif = min(min_dif, dif)
        dif_sum += dif
    return min_dif, max_dif, dif_sum

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
        res = np.array(res, dtype=np.float)
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

            src_data = get_num(src.split(":")[1], "int").reshape(s0,s1,s2)
            pos_data = get_num(pos.split(":")[1], "int").reshape(s0,s1,s2)
            sent_data = get_num(sent.split(":")[1], "int").reshape(s0,s1,s2)
            mask_data = get_num(mask.split(":")[1], "int").reshape(s0,s1,s2)
            # convert mask (-1, -1, 1) to (-1, 1)
            mask = []
            for i in range(s0):
                sum = 0
                for j in range(s1):
                    for k in range(s2):
                        sum += mask_data[i][j][k]
                mask.append(sum)
            mask = np.array(mask, dtype=np.int32).reshape(-1, 1)

            aside_shape = get_num(aside1.split(":")[0], "int")
            s0 = aside_shape[0]
            s1 = aside_shape[1]
            s2 = aside_shape[2]

            aside1_data = get_num(aside1.split(":")[1], "int").reshape(s0,s1,s2)
            aside2_data = get_num(aside2.split(":")[1], "int").reshape(s0,s1,s2)
            aside3_data = get_num(aside3.split(":")[1], "int").reshape(s0,s1,s2)
            aside4_data = get_num(aside4.split(":")[1], "int").reshape(s0,s1,s2)
            aside5_data = get_num(aside5.split(":")[1], "int").reshape(s0,s1,s2)
            aside6_data = get_num(aside6.split(":")[1], "int").reshape(s0,s1,s2)
            aside7_data = get_num(aside7.split(":")[1], "int").reshape(s0,s1,s2)
            aside8_data = get_num(aside8.split(":")[1], "int").reshape(s0,s1,s2)

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

            res.append([src_data, pos_data, sent_data, mask, aside1_data, aside2_data, aside3_data, aside4_data, aside5_data, aside6_data, aside7_data, aside8_data])
    return res

def main(args):
    # init infer_helper
    plan_name = args.plan_name
    file_name = args.input_file
    infer_helper = trt_helper.InferHelper(plan_name, logger)

    # 1 batch test data
    # src = [1,12,13,1557,40574,40997,22553,2,1571,40574,1569,42562,1557,40997,22553,1886,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # pos = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # sent = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # mask = [18]
    # aside = [1]
    # src = np.array(src,dtype=np.int32).reshape(1,128,1)
    # pos = np.array(pos,dtype=np.int32).reshape(1,128,1)
    # sent = np.array(sent,dtype=np.int32).reshape(1,128,1)
    # mask = np.array(mask,dtype=np.float32).reshape(1,128, 1)
    # aside = np.array(aside,dtype=np.int32).reshape(1,1,1)
    # print(src.shape, src.dtype)
    # print(pos.shape, pos.dtype)
    # print(sent.shape, sent.dtype)
    # print(mask.shape, mask.dtype)
    # print(aside.shape, aside.dtype)
    # inputs = [src, sent, pos, mask, aside, aside, aside, aside, aside, aside, aside, aside]

    # multi batch
    inputs = load_data(file_name)

    # load onnx baseline
    onnx_baseline = []
    with open("./onnx_baseline.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            data = get_num(line, "float")
            onnx_baseline.append(data)

    # test_num = len(inputs)
    test_num = 1

    # trt infer and compare with onnx baseline
    max_dif = 0
    min_dif = 1
    dif_sum = 0
    total_num = 0
    for i in range(test_num):
        output = infer_helper.infer(inputs[i], True)
        b = onnx_baseline[i].shape[0]
        print("infer and comparing case", i)
        mindif, maxdif, dif = cal_dif(onnx_baseline[i].reshape(b), output[0].reshape(b), b)
        max_dif = max(max_dif, maxdif)
        min_dif = min(min_dif, mindif)
        dif_sum += dif
        total_num += b
    print("min_dif:", min_dif, " max_dif:", max_dif, " avg_dif:", dif_sum / total_num)

    # speed test
    # warm up
    # output = infer_helper.infer(inputs[0], True)

    # start = time.perf_counter()
    # for i in range(test_num):
    #     output = infer_helper.infer(inputs[i], True)
    # end = time.perf_counter()

    # print("avg_time", (end-start) / test_num * 1000, "ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ernie trt model test")

    parser.add_argument("-p", "--plan_name", required=True, help="The trt plan file path.")
    parser.add_argument("-i", "--input_file", required=True, help="The onnx baseline file path.")

    args = parser.parse_args()
    main(args)
