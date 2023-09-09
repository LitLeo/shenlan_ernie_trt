#include <sys/time.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
// #include "paddle_inference_api.h"
#include "trt_helper.h"

// using paddle_infer::Config;
// using paddle_infer::Predictor;
// using paddle_infer::CreatePredictor;

static const int MAX_SEQ = 128;

// struct sample{
//     std::string qid;
//     std::string label;
//     std::vector<int> shape_info_0;
//     std::vector<int64_t> i0;
//     std::vector<int> shape_info_1;
//     std::vector<int64_t> i1;
//     std::vector<int> shape_info_2;
//     std::vector<int64_t> i2;
//     std::vector<int> shape_info_3;
//     std::vector<float> i3;
//     std::vector<int> shape_info_4;
//     std::vector<int64_t> i4;
//     std::vector<int> shape_info_5;
//     std::vector<int64_t> i5;
//     std::vector<int> shape_info_6;
//     std::vector<int64_t> i6;
//     std::vector<int> shape_info_7;
//     std::vector<int64_t> i7;
//     std::vector<int> shape_info_8;
//     std::vector<int64_t> i8;
//     std::vector<int> shape_info_9;
//     std::vector<int64_t> i9;
//     std::vector<int> shape_info_10;
//     std::vector<int64_t> i10;
//     std::vector<int> shape_info_11;
//     std::vector<int64_t> i11;
//     std::vector<float> out_data;
//     uint64_t timestamp;
// };

void split_string(const std::string& str,
                  const std::string& delimiter,
                  std::vector<std::string>& fields) {
    size_t pos = 0;
    size_t start = 0;
    size_t length = str.length();
    std::string token;
    while ((pos = str.find(delimiter, start)) != std::string::npos && start < length) {
        token = str.substr(start, pos - start);
        fields.push_back(token);
        start += delimiter.length() + token.length(); 
    }
    if (start <= length - 1) {
        token = str.substr(start);
        fields.push_back(token);
    }
}

void field2vec(const std::string& input_str,
               bool padding,
               std::vector<int>* shape_info,
               std::vector<int64_t>* i64_vec,
               std::vector<float>* f_vec = nullptr) {
    std::vector<std::string> i_f;
    split_string(input_str, ":", i_f);
    std::vector<std::string> i_v;
    split_string(i_f[1], " ", i_v);
    std::vector<std::string> s_f;
    split_string(i_f[0], " ", s_f);
    for (auto& f : s_f) {
        shape_info->push_back(std::stoi(f));
    }
    int batch_size = shape_info->at(0);
    int seq_len = shape_info->at(1);
    if (i64_vec) {
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                i64_vec->push_back(std::stoll(i_v[i * seq_len + j]));
            }
            // padding to MAX_SEQ_LEN
            for (int j = 0; padding && j < MAX_SEQ - seq_len; ++j) {
                i64_vec->push_back(0);
            }
        }
    } else {
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                f_vec->push_back(std::stof(i_v[i * seq_len + j]));
            }
            // padding to MAX_SEQ_LEN
            for (int j = 0; padding && j < MAX_SEQ - seq_len; ++ j) {
                f_vec->push_back(0);
            }
        }
    }
    if (padding) {
        (*shape_info)[1] = MAX_SEQ;
    }
}

void line2sample(const std::string& line, sample* sout) {
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

// TrtHepler::TrtHepler
// std::shared_ptr<Predictor> InitPredictor(const std::string& model_file,
//                                          const std::string& params_file) {
//   Config config;
//   config.SetModel(model_file, params_file);
//   config.EnableUseGpu(100, 0);
//   // Open the memory optim.
//   config.EnableMemoryOptim();
//   // config.Exp_DisableTensorRtOPs({"concat"});
//   // only kHalf supported
//   // config.EnableTensorRtEngine(1 << 30, 10, 5, Config::Precision::kFloat32, false,
//   //                            true);
//   // dynamic shape
//   // config.SetTRTDynamicShapeInfo(min_input_shape, max_input_shape,
//   //                               opt_input_shape);
//   return CreatePredictor(config);
// }

// void run(Predictor *predictor, sample& s) {
//   auto input_names = predictor->GetInputNames();

//   auto set_feature_input = [&predictor](
//                             const std::string& input_name,
//                             const std::vector<int>& shape_info,
//                             const int64_t* data_ptr) {
//       auto cur_input_t = predictor->GetInputHandle(input_name);
//       cur_input_t->Reshape(shape_info);
//       cur_input_t->CopyFromCpu(data_ptr);
//   };
//   // first input
//   set_feature_input(input_names[0], s.shape_info_0, s.i0.data());
//   // second input
//   set_feature_input(input_names[1], s.shape_info_1, s.i1.data());
//   // third input
//   set_feature_input(input_names[2], s.shape_info_2, s.i2.data());
//   // fourth input
//   auto input_t4 = predictor->GetInputHandle(input_names[3]);
//   input_t4->Reshape(s.shape_info_3);
//   input_t4->CopyFromCpu(s.i3.data());
//   // fifth input
//   set_feature_input(input_names[4], s.shape_info_4, s.i4.data());
//   set_feature_input(input_names[5], s.shape_info_5, s.i5.data());
//   set_feature_input(input_names[6], s.shape_info_6, s.i6.data());
//   set_feature_input(input_names[7], s.shape_info_7, s.i7.data());
//   set_feature_input(input_names[8], s.shape_info_8, s.i8.data());
//   set_feature_input(input_names[9], s.shape_info_9, s.i9.data());
//   set_feature_input(input_names[10], s.shape_info_10, s.i10.data());
//   set_feature_input(input_names[11], s.shape_info_11, s.i11.data());

//   predictor->Run();

//   auto output_names = predictor->GetOutputNames();
//   auto output_t = predictor->GetOutputHandle(output_names[0]);
//   std::vector<int> output_shape = output_t->shape();
//   int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
//                                 std::multiplies<int>());
//   s.out_data.resize(out_num);
//   output_t->CopyToCpu(s.out_data.data());
//   struct timeval tv;
//   gettimeofday(&tv, NULL);
//   s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;
//   return;
// }

int main(int argc, char *argv[]) {
  // init
  std::string model_name = argv[1];
  
  // std::string para_file = argv[2];
  std::string model_para_file = argv[2];
  std::cout << model_para_file << std::endl;
  // auto predictor = InitPredictor(model_name, para_file);
  auto trt_helper = new TrtHepler(model_para_file, 0);
  // preprocess
  std::string aline;
  std::ifstream ifs;
  ifs.open(argv[3], std::ios::in);
  std::ofstream ofs;
  ofs.open(argv[4], std::ios::out);
  std::vector<sample> sample_vec;
  while (std::getline(ifs, aline)) {
      sample s;
      line2sample(aline, &s);
      sample_vec.push_back(s);
  }

  // inference
  for (auto& s : sample_vec) {
      // //run(predictor.get(), s);
      trt_helper->Forward(s);
  }
 
  // postprocess
  for (auto& s : sample_vec) {
      std::ostringstream oss;
      oss << s.qid << "\t";
      oss << s.label << "\t";
      for (int i = 0; i < s.out_data.size(); ++i) {
          oss << s.out_data[i];
          if (i == s.out_data.size() - 1) {
              oss << "\t";
          } else {
              oss << ","; 
          }
      }
      oss << s.timestamp << "\n";
      ofs.write(oss.str().c_str(), oss.str().length());
  }
  ofs.close();
  ifs.close();
  return 0;
}
