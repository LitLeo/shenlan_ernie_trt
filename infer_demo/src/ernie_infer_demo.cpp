#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "trt_helper.h"

using namespace nvinfer1::plugin;

using namespace std;

static const int MAX_SEQ = 128;

void split_string(const std::string &str, const std::string &delimiter,
                  std::vector<std::string> &fields) {
  size_t pos = 0;
  size_t start = 0;
  size_t length = str.length();
  std::string token;
  while ((pos = str.find(delimiter, start)) != std::string::npos &&
         start < length) {
    token = str.substr(start, pos - start);
    fields.push_back(token);
    start += delimiter.length() + token.length();
  }
  if (start <= length - 1) {
    token = str.substr(start);
    fields.push_back(token);
  }
}

void field2vec(const std::string &input_str, bool padding,
               std::vector<int> *shape_info, std::vector<int> *i64_vec,
               std::vector<int> *f_vec = nullptr) {
  std::vector<std::string> i_f;
  split_string(input_str, ":", i_f);
  std::vector<std::string> i_v;
  split_string(i_f[1], " ", i_v);
  std::vector<std::string> s_f;
  split_string(i_f[0], " ", s_f);
  for (auto &f : s_f) {
    shape_info->push_back(std::stoi(f));
  }
  int batch_size = shape_info->at(0);
  int seq_len = shape_info->at(1);
  int padding_seq_len = seq_len;
  if (padding_seq_len >= 64)
    padding_seq_len = (padding_seq_len + 31) / 32 * 32;
  if (i64_vec) {
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < seq_len; ++j) {
        i64_vec->push_back(std::stoi(i_v[i * seq_len + j]));
      }
      // padding to MAX_SEQ_LEN
      for (int j = 0; padding && j < padding_seq_len - seq_len; ++j) {
        i64_vec->push_back(0);
      }
    }
  } else {
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < seq_len; ++j) {
        auto tmp = static_cast<int>(std::stof(i_v[i * seq_len + j]));
        f_vec->push_back(tmp);
      }
      // padding to MAX_SEQ_LEN
      for (int j = 0; padding && j < padding_seq_len - seq_len; ++j) {
        f_vec->push_back(0);
      }
    }
  }
  // if (padding) {
  // (*shape_info)[1] = MAX_SEQ;
  // }
}

void line2sample(const std::string &line, sample *sout) {
  std::vector<std::string> fields;
  split_string(line, ";", fields);
  assert(fields.size() == 14);
  // parse qid
  std::vector<std::string> qid_f;
  split_string(fields[0], ":", qid_f);
  sout->qid = qid_f[1];
  // Parse label
  std::vector<std::string> label_f;
  split_string(fields[1], ":", label_f);
  sout->label = label_f[1];
  // Parse input field
  field2vec(fields[2], true, &(sout->shape_info_0), &(sout->i0));
  field2vec(fields[3], true, &(sout->shape_info_1), &(sout->i1));
  field2vec(fields[4], true, &(sout->shape_info_2), &(sout->i2));
  field2vec(fields[5], true, &(sout->shape_info_3), nullptr, &(sout->i3));
  field2vec(fields[6], false, &(sout->shape_info_4), &(sout->i4));
  field2vec(fields[7], false, &(sout->shape_info_5), &(sout->i5));
  field2vec(fields[8], false, &(sout->shape_info_6), &(sout->i6));
  field2vec(fields[9], false, &(sout->shape_info_7), &(sout->i7));
  field2vec(fields[10], false, &(sout->shape_info_8), &(sout->i8));
  field2vec(fields[11], false, &(sout->shape_info_9), &(sout->i9));
  field2vec(fields[12], false, &(sout->shape_info_10), &(sout->i10));
  field2vec(fields[13], false, &(sout->shape_info_11), &(sout->i11));
  return;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    cout << "ERROR: argc != 4 && argc != 11" << endl;
    return -1;
  }
  // nvinfer1::plugin::hello();
  // init
  int argc_idx = 1;
  string model_file = argv[argc_idx++];
  std::string test_file = argv[argc_idx++];
  std::string out_file = argv[argc_idx++];

  // std::string para_file = argv[2];
  // std::string model_para_file = argv[2];
  // std::cout << model_para_file << std::endl;
  // auto predictor = InitPredictor(model_name, para_file);
  auto trt_engine = new TrtEngine(model_file, 0);
  // auto trt_context = new TrtContext(trt_engine, 0);

  auto trt_context1 = new TrtContext(trt_engine, 0);
  auto trt_context2 = new TrtContext(trt_engine, 1);
  auto trt_context3 = new TrtContext(trt_engine, 2);
  auto trt_context4 = new TrtContext(trt_engine, 3);

  // preprocess
  std::string aline;
  std::ifstream ifs;
  ifs.open(test_file, std::ios::in);
  std::ofstream ofs;
  ofs.open(out_file, std::ios::out);
  std::vector<sample> sample_vec;
  while (std::getline(ifs, aline)) {
    sample s;
    line2sample(aline, &s);
    sample_vec.push_back(s);
  }

  // inference
  int idx = 0;
  for (auto &s : sample_vec) {
    int seq_len = s.shape_info_0[1];
    if (seq_len < 64) {
      trt_context1->Forward(s);
    } else if (seq_len == 64) {
      trt_context2->Forward(s);
    } else if (seq_len <= 96) {
      trt_context3->Forward(s);
    } else {
      trt_context4->Forward(s);
    }
    // trt_context->Forward(s);
    idx++;
    // if (idx == 5)
    // break;
    // if (idx == 2)
    //     break;
  }

  // postprocess
  for (auto &s : sample_vec) {
    std::ostringstream oss;
    oss << s.qid << "\t";
    oss << s.label << "\t";
    for (int i = 0; i < s.out_data.size(); ++i) {
      oss << s.out_data[i];
      // cout << s.out_data[i] << " ";
      if (i == s.out_data.size() - 1) {
        oss << "\t";
      } else {
        oss << ",";
      }
    }
    oss << s.timestamp << "\n";
    ofs.write(oss.str().c_str(), oss.str().length());
    // break;
  }
  ofs.close();
  ifs.close();

  delete trt_engine;
  // delete trt_context;
  delete trt_context1;
  delete trt_context2;
  delete trt_context3;
  delete trt_context4;

  return 0;
}
