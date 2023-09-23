#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np


class ErnieCalibrator(trt.IInt8LegacyCalibrator):
    def __init__(self, file_path, cache_file, batch_size, max_seq_length, num_inputs):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8LegacyCalibrator.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        # self.data = dp.read_squad_json(squad_json)

        src = np.fromfile("./src/calib_data/src.npy", dtype=np.int32).reshape(10009, 128, 1)[:num_inputs, :, :]
        pos = np.fromfile("./src/calib_data/pos.npy", dtype=np.int32).reshape(10009, 128, 1)[:num_inputs, :, :]
        sent = np.fromfile("./src/calib_data/sent.npy", dtype=np.int32).reshape(10009, 128, 1)[:num_inputs, :, :]
        mask = np.fromfile("./src/calib_data/mask.npy", dtype=np.int32).reshape(10009, 1)[:num_inputs, :]
        aside1 = np.fromfile("./src/calib_data/aside1.npy", dtype=np.int32).reshape(10009, 1)[:num_inputs, :]
        aside2 = np.fromfile("./src/calib_data/aside2.npy", dtype=np.int32).reshape(10009, 1)[:num_inputs, :]
        aside3 = np.fromfile("./src/calib_data/aside3.npy", dtype=np.int32).reshape(10009, 1)[:num_inputs, :]
        aside4 = np.fromfile("./src/calib_data/aside4.npy", dtype=np.int32).reshape(10009, 1)[:num_inputs, :]
        aside5 = np.fromfile("./src/calib_data/aside5.npy", dtype=np.int32).reshape(10009, 1)[:num_inputs, :]
        aside6 = np.fromfile("./src/calib_data/aside6.npy", dtype=np.int32).reshape(10009, 1)[:num_inputs, :]
        aside7 = np.fromfile("./src/calib_data/aside7.npy", dtype=np.int32).reshape(10009, 1)[:num_inputs, :]
        aside8 = np.fromfile("./src/calib_data/aside8.npy", dtype=np.int32).reshape(10009, 1)[:num_inputs, :]

        # feat_list = []
        # feat_len_list = []
        # for i in range(1, num_inputs + 1):
        #     # padding?
        #     feat_file = file_path + '/feat' + str(i) + '.npy'
        #     feat = np.load(feat_file)

        #     feat_len = np.array([[0]], dtype=np.int32)
        #     feat_len[0][0] = feat.shape[1]

        #     feat_list.append(feat)
        #     feat_len_list.append(feat_len)

        self.data = [src, pos, sent, mask, aside1, aside2, aside3, aside4, aside5, aside6, aside7, aside8]
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.current_index = 0
        self.num_inputs = num_inputs
        # self.tokenizer = tokenization.BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
        # self.doc_stride = 128
        # self.max_query_length = 64

        # Allocate enough memory for a whole batch.
        self.device_inputs = [cuda.mem_alloc(self.max_seq_length * trt.float32.itemsize * self.batch_size),
                              cuda.mem_alloc(self.max_seq_length * trt.float32.itemsize * self.batch_size),
                              cuda.mem_alloc(self.max_seq_length * trt.float32.itemsize * self.batch_size),
                              cuda.mem_alloc(trt.float32.itemsize * self.batch_size),
                              cuda.mem_alloc(trt.float32.itemsize * self.batch_size),
                              cuda.mem_alloc(trt.float32.itemsize * self.batch_size),
                              cuda.mem_alloc(trt.float32.itemsize * self.batch_size),
                              cuda.mem_alloc(trt.float32.itemsize * self.batch_size),
                              cuda.mem_alloc(trt.float32.itemsize * self.batch_size),
                              cuda.mem_alloc(trt.float32.itemsize * self.batch_size),
                              cuda.mem_alloc(trt.float32.itemsize * self.batch_size),
                              cuda.mem_alloc(trt.float32.itemsize * self.batch_size)]

    def free(self):
        for dinput in self.device_inputs:
            dinput.free()

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.num_inputs:
            print("Calibrating index {:} batch size {:} exceed max input limit {:} sentences".format(self.current_index,
                self.batch_size, self.num_inputs))
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % self.batch_size == 0:
            print("Calibrating batch {:}, containing {:} sentences".format(current_batch, self.batch_size))

        cur_src = self.data[0][self.current_index:self.current_index+self.batch_size, :, :]
        cur_pos = self.data[1][self.current_index:self.current_index+self.batch_size, :, :]
        cur_sent = self.data[2][self.current_index:self.current_index+self.batch_size, :, :]
        cur_mask = self.data[3][self.current_index:self.current_index+self.batch_size, :]
        cur_aside1 = self.data[4][self.current_index:self.current_index+self.batch_size, :]
        cur_aside2 = self.data[5][self.current_index:self.current_index+self.batch_size, :]
        cur_aside3 = self.data[6][self.current_index:self.current_index+self.batch_size, :]
        cur_aside4 = self.data[7][self.current_index:self.current_index+self.batch_size, :]
        cur_aside5 = self.data[8][self.current_index:self.current_index+self.batch_size, :]
        cur_aside6 = self.data[9][self.current_index:self.current_index+self.batch_size, :]
        cur_aside7 = self.data[10][self.current_index:self.current_index+self.batch_size, :]
        cur_aside8 = self.data[11][self.current_index:self.current_index+self.batch_size, :]

        cuda.memcpy_htod(self.device_inputs[0], cur_src.ravel())
        cuda.memcpy_htod(self.device_inputs[1], cur_pos.ravel())
        cuda.memcpy_htod(self.device_inputs[2], cur_sent.ravel())
        cuda.memcpy_htod(self.device_inputs[3], cur_mask.ravel())
        cuda.memcpy_htod(self.device_inputs[4], cur_aside1.ravel())
        cuda.memcpy_htod(self.device_inputs[5], cur_aside2.ravel())
        cuda.memcpy_htod(self.device_inputs[6], cur_aside3.ravel())
        cuda.memcpy_htod(self.device_inputs[7], cur_aside4.ravel())
        cuda.memcpy_htod(self.device_inputs[8], cur_aside5.ravel())
        cuda.memcpy_htod(self.device_inputs[9], cur_aside6.ravel())
        cuda.memcpy_htod(self.device_inputs[10], cur_aside7.ravel())
        cuda.memcpy_htod(self.device_inputs[11], cur_aside8.ravel())

        # feat = []
        # feat_len = []
        # for i in range(self.batch_size):
        #     idx = self.current_index + i
        #     cur_feat = self.data[0][idx][0]
        #     cur_feat_len = self.data[1][idx][0]
        #     if len(feat) and len(feat_len):
        #         feat = np.concatenate((feat, cur_feat))
        #         feat_len = np.concatenate((feat_len, cur_feat_len))
        #         # input_ids = np.concatenate((input_ids, cur_input_ids))
        #         # segment_ids = np.concatenate((segment_ids, cur_segment_ids))
        #         # input_mask = np.concatenate((input_mask, cur_mask))
        #     else:
        #         feat = cur_feat
        #         feat_len = cur_feat_len
        #         # input_ids = cur_input_ids
        #         # segment_ids = cur_segment_ids
        #         # input_mask = cur_mask

        # cuda.memcpy_htod(self.device_inputs[0], feat.ravel())
        # cuda.memcpy_htod(self.device_inputs[1], feat_len.ravel())
        # # cuda.memcpy_htod(self.device_inputs[0], input_ids.ravel())
        # # cuda.memcpy_htod(self.device_inputs[1], segment_ids.ravel())
        # # cuda.memcpy_htod(self.device_inputs[2], input_mask.ravel())

        self.current_index += self.batch_size
        return self.device_inputs

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            os.fsync(f)

    def get_quantile(self):
        return 0.9999

    def get_regression_cutoff(self):
        return 1.0

    def read_histogram_cache(self, length):
        return None

    def write_histogram_cache(self, ptr, length):
        return None
