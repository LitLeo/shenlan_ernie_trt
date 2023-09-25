#include "trt_helper.h"

#include <fstream>
#include <sstream>
#include <string>

using namespace std;

// BEGIN_LIB_NAMESPACE {

TrtLogger::TrtLogger(nvinfer1::ILogger::Severity level) : level_(level) {}

nvinfer1::ILogger &TrtLogger::getTRTLogger() { return *this; }

// trt logger
void TrtLogger::log(Severity severity, const char *msg) noexcept {
  if (severity > level_) {
    return;
  }

  switch (severity) {
  case Severity::kINTERNAL_ERROR:
    std::cout << "[TRT] " << std::string(msg) << std::endl;
    break;
  case Severity::kERROR:
    std::cout << "[TRT] " << std::string(msg) << std::endl;
    break;
  case Severity::kWARNING:
    std::cout << "[TRT] " << std::string(msg) << std::endl;
    break;
  case Severity::kINFO:
    std::cout << "[TRT] " << std::string(msg) << std::endl;
    break;
  case Severity::kVERBOSE:
    std::cout << "[TRT] " << std::string(msg) << std::endl;
  }
}

TrtEngine::TrtEngine(std::string model_param, int dev_id_)
    : dev_id_(dev_id_), _model_param(model_param) {
  std::ifstream t(_model_param);  // string pth
  std::stringstream buffer;
  buffer << t.rdbuf();
  std::string contents(buffer.str());

  CUDA_CHECK(cudaSetDevice(dev_id_));

  auto runtime =
      MakeUnique(nvinfer1::createInferRuntime(trt_logger.getTRTLogger()));
  auto e = runtime->deserializeCudaEngine((void *)contents.c_str(),
                                          contents.size(), nullptr);
  engine_ = MakeShared(e);

  initLibNvInferPlugins(&trt_logger, "");
}

TrtContext::TrtContext(TrtEngine *trt_engine, int profile_idx) {
  engine_ = trt_engine->engine_;
  dev_id_ = trt_engine->dev_id_;
  CUDA_CHECK(cudaSetDevice(dev_id_));
  CUDA_CHECK(cudaStreamCreate(&cuda_stream_));

  context_ = MakeShared(engine_->createExecutionContext());
  // context_->setOptimizationProfileAsync(profile_idx, cuda_stream_);
  context_->setOptimizationProfile(profile_idx);

  start_binding_idx_ = profile_idx * engine_->getNbBindings() /
                       engine_->getNbOptimizationProfiles();
  auto min_profile = engine_->getProfileDimensions(
      start_binding_idx_, profile_idx, nvinfer1::OptProfileSelector::kMIN);
  auto max_profile = engine_->getProfileDimensions(
      start_binding_idx_, profile_idx, nvinfer1::OptProfileSelector::kMAX);

  max_batch_ = max_profile.d[0];
  max_seq_len_ = max_profile.d[1];

  min_batch_ = min_profile.d[0];
  min_seq_len_ = min_profile.d[1];

  int whole_input = 3 * max_batch_ * max_seq_len_ * sizeof(int) +
                    (max_batch_ * sizeof(int) + 127) / 128 * 128 * 8;
  int output_max_size = (max_batch_ * sizeof(int) + 127) / 128 * 128;
  int whole_size = whole_input + output_max_size;

  cudaMalloc(&d_buffer_, whole_size);
  cudaMallocHost(&h_buffer_, whole_size);
  // h_buffer_ = (char*)malloc(whole_size);

  device_bindings_.resize(engine_->getNbBindings());
  for (int i = 0; i < engine_->getNbBindings(); i++) {
    device_bindings_[i] = d_buffer_;
  }

  cout << "start_binding_idx_ = " << start_binding_idx_ << endl;
}

int TrtContext::Forward(sample &s) {
  cudaSetDevice(dev_id_);
  int idx = 0;

  // std::vector<int> mask_idx;
  // auto si3_ptr = s.i3.data();
  // for (int i = 0; i < s.shape_info_3[0]; i++) {
  // int mask_len = 0;
  // for (int j = 0; j < s.shape_info_3[1]; j++) {
  // if (si3_ptr[j] != 0.0) {
  // mask_len++;
  //} else break;
  //}
  // si3_ptr = si3_ptr + s.shape_info_3[1];
  // mask_idx.push_back(mask_len);
  // cout << mask_len << " ";
  //}
  // vector<int> s3_shape = {s.shape_info_3[0], 1};
  int data_len = 1;

  int batch = s.shape_info_0[0];
  int seq_len = s.shape_info_0[1];
  int padding_seq_len = seq_len;

  if (padding_seq_len >= 64) {
    // 64, 96, 128
    padding_seq_len = (padding_seq_len + 31) / 32 * 32;
  }

  int input_bytes = batch * padding_seq_len * sizeof(int);
  int aside_input_bytes = batch * sizeof(int);

  int align_input_bytes = (input_bytes + 127) / 128 * 128;
  int align_aside_input_bytes = (aside_input_bytes + 127) / 128 * 128;

  int s0_offset = 0;  // src_ids
  int s2_offset = align_input_bytes;  // sent_ids
  int s3_offset = s2_offset + align_input_bytes;  // input mask

  int s4_offset = s3_offset + align_input_bytes;
  int s5_offset = s4_offset + align_aside_input_bytes;
  int s6_offset = s5_offset + align_aside_input_bytes;
  int s7_offset = s6_offset + align_aside_input_bytes;
  int s8_offset = s7_offset + align_aside_input_bytes;
  int s9_offset = s8_offset + align_aside_input_bytes;
  int s10_offset = s9_offset + align_aside_input_bytes;
  int s11_offset = s10_offset + align_aside_input_bytes;

  int out_offset = s11_offset + align_aside_input_bytes;

  memcpy(h_buffer_ + s0_offset, s.i0.data(), input_bytes);
  memcpy(h_buffer_ + s2_offset, s.i2.data(), input_bytes);
  memcpy(h_buffer_ + s3_offset, s.i3.data(), input_bytes);

  memcpy(h_buffer_ + s4_offset, s.i4.data(), aside_input_bytes);
  memcpy(h_buffer_ + s5_offset, s.i5.data(), aside_input_bytes);
  memcpy(h_buffer_ + s6_offset, s.i6.data(), aside_input_bytes);
  memcpy(h_buffer_ + s7_offset, s.i7.data(), aside_input_bytes);
  memcpy(h_buffer_ + s8_offset, s.i8.data(), aside_input_bytes);
  memcpy(h_buffer_ + s9_offset, s.i9.data(), aside_input_bytes);
  memcpy(h_buffer_ + s10_offset, s.i10.data(), aside_input_bytes);
  memcpy(h_buffer_ + s11_offset, s.i11.data(), aside_input_bytes);

  cudaEvent_t start, stop;
  float elapsed_time = 0.0;
  cudaStreamSynchronize(cuda_stream_);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  int ret =
      cudaMemcpyAsync(d_buffer_, h_buffer_, s11_offset + aside_input_bytes,
                      cudaMemcpyHostToDevice, cuda_stream_);
  if (ret)
    printf("CpuToDeviceTpl error\n");

  vector<int> input_dim = {batch, padding_seq_len, 1};

  int binding_idx = start_binding_idx_;
  std::vector<std::vector<int>> input_dims = {
      input_dim,      input_dim,       input_dim,      s.shape_info_4,
      s.shape_info_5, s.shape_info_6,  s.shape_info_7, s.shape_info_8,
      s.shape_info_9, s.shape_info_10, s.shape_info_11};
  // set device_bindings_ and setBindingDimensions
  for (size_t i = 0; i < input_dims.size(); i++) {
    std::vector<int> dims_vec = input_dims[i];
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = static_cast<int>(dims_vec.size());
    memcpy(trt_dims.d, dims_vec.data(), sizeof(int) * trt_dims.nbDims);
    context_->setBindingDimensions(binding_idx, trt_dims);
    binding_idx++;
  }

  if (!context_->allInputDimensionsSpecified()) {
    // gLogFatal << "context_->allInputDimensionsSpecified() error";
    std::cout << ("context_->allInputDimensionsSpecified() error") << std::endl;
    assert(0);
  }

  // set the input dim
  binding_idx = start_binding_idx_;
  device_bindings_[binding_idx++] = d_buffer_;
  device_bindings_[binding_idx++] = d_buffer_ + s2_offset;
  device_bindings_[binding_idx++] = d_buffer_ + s3_offset;
  device_bindings_[binding_idx++] = d_buffer_ + s4_offset;
  device_bindings_[binding_idx++] = d_buffer_ + s5_offset;
  device_bindings_[binding_idx++] = d_buffer_ + s6_offset;
  device_bindings_[binding_idx++] = d_buffer_ + s7_offset;
  device_bindings_[binding_idx++] = d_buffer_ + s8_offset;
  device_bindings_[binding_idx++] = d_buffer_ + s9_offset;
  device_bindings_[binding_idx++] = d_buffer_ + s10_offset;
  device_bindings_[binding_idx++] = d_buffer_ + s11_offset;
  device_bindings_[binding_idx++] = d_buffer_ + out_offset;

  void *device_bindings[12] = {d_buffer_,
                               d_buffer_ + s2_offset,
                               d_buffer_ + s3_offset,
                               d_buffer_ + s4_offset,
                               d_buffer_ + s5_offset,
                               d_buffer_ + s6_offset,
                               d_buffer_ + s7_offset,
                               d_buffer_ + s8_offset,
                               d_buffer_ + s9_offset,
                               d_buffer_ + s10_offset,
                               d_buffer_ + s11_offset,
                               d_buffer_ + out_offset};
  // printf("before enqueue\n");
  ret = context_->enqueueV2((void **)device_bindings_.data(), cuda_stream_,
                            nullptr);
  if (!ret) {
    // gLogError << " context_->enqueueV2 failed!";
    std::cout << ("context_->enqueueV2 failed!") << std::endl;
    assert(0);
    return -100;
  }

  // cudaStreamSynchronize(cuda_stream_);
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);
  s.out_data.resize(batch);
  cudaMemcpyAsync(s.out_data.data(), d_buffer_ + out_offset,
                  batch * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream_);
  cudaStreamSynchronize(cuda_stream_);

  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&elapsed_time, start, stop);
  //// auto dim_str = Dims2String(_inputs_dims[0]);
  //cout << "batch=" << batch << ", seq_len = " << seq_len
       //<< ", padding_seq_len = " << padding_seq_len << ", time=" << elapsed_time
       //<< " ms" << endl;
  //cudaEventDestroy(start);
  //cudaEventDestroy(stop);

  // 计算当次推理结束的时间戳
  struct timeval tv;
  gettimeofday(&tv, NULL);
  s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;
}

TrtContext::~TrtContext() {
  CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
  cudaFree(d_buffer_);
  cudaFreeHost(h_buffer_);
}

// } // BEGIN_LIB_NAMESPACE
