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

  initLibNvInferPlugins(&trt_logger.getTRTLogger(), "");
  auto runtime =
      MakeUnique(nvinfer1::createInferRuntime(trt_logger.getTRTLogger()));
  auto e = runtime->deserializeCudaEngine((void *)contents.c_str(),
                                          contents.size(), nullptr);
  engine_ = MakeShared(e);

  auto device_memory_size = engine_->getDeviceMemorySize();
  CUDA_CHECK(cudaMalloc(&device_memory_, device_memory_size));

  cout << "getNbIOTensors:" << engine_->getNbIOTensors() << endl;
}

constexpr size_t kAlignment = 128;
constexpr int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}
constexpr int AlignTo(int a, int b = kAlignment) {
  return ceildiv(a, b) * b;
}

std::vector<char*> TrtContext::s_device_bindings_;

TrtContext::TrtContext(TrtEngine *trt_engine, int profile_idx) {
  profile_idx_ = profile_idx;
  engine_ = trt_engine->engine_;
  dev_id_ = trt_engine->dev_id_;
  CUDA_CHECK(cudaSetDevice(dev_id_));
  CUDA_CHECK(cudaStreamCreate(&cuda_stream_));

  context_ = MakeShared(engine_->createExecutionContextWithoutDeviceMemory());
  context_->setDeviceMemory(trt_engine->device_memory_);
  // context_->setOptimizationProfileAsync(profile_idx, cuda_stream_);
  context_->setOptimizationProfile(profile_idx);

  start_binding_idx_ = profile_idx * engine_->getNbBindings() /
                       engine_->getNbOptimizationProfiles();
  auto max_profile = engine_->getProfileDimensions(
      start_binding_idx_, profile_idx, nvinfer1::OptProfileSelector::kMAX);

  max_batch_ = max_profile.d[0];
  max_seq_len_ = max_profile.d[1];

  // 4 inputs: [B, S]
  align_input_bytes_ = AlignTo(max_batch_ * max_seq_len_ * sizeof(int));
  // 8 aside inputs and 1 output: [B]
  align_aside_input_bytes_ = AlignTo(max_batch_ * sizeof(int));

  whole_bytes_ = align_input_bytes_ * 4 + align_aside_input_bytes_ * 9;

  CUDA_CHECK(cudaMalloc(&d_buffer_, whole_bytes_));
  CUDA_CHECK(cudaMallocHost(&h_buffer_, whole_bytes_));

  auto h_buffer_ptr = h_buffer_;
  auto d_buffer_ptr = d_buffer_;

  device_bindings_.resize(engine_->getNbBindings());
  for (size_t i = 0; i < device_bindings_.size(); i++) {
    device_bindings_[i] = d_buffer_ptr;
  }

  // 4 inputs: [B, S]
  int b_i = 0;

  while (b_i < 4) {
    device_bindings_[start_binding_idx_ + b_i] = d_buffer_ptr;
    host_bindings_.push_back(h_buffer_ptr);

    h_buffer_ptr += align_input_bytes_;
    d_buffer_ptr += align_input_bytes_;

    b_i ++;
  }

  while (b_i < 13) {
    device_bindings_[start_binding_idx_ + b_i] = d_buffer_ptr;
    host_bindings_.push_back(h_buffer_ptr);

    h_buffer_ptr += align_aside_input_bytes_;
    d_buffer_ptr += align_aside_input_bytes_;

    b_i ++;
  }

  vector<int> input_dim = {max_batch_, max_seq_len_, 1};
  vector<int> aside_input_dim = {max_batch_, 1, 1};

  int binding_idx = start_binding_idx_;
  std::vector<std::vector<int>> input_dims = {
      input_dim, input_dim, input_dim, input_dim,
      aside_input_dim, aside_input_dim, aside_input_dim, aside_input_dim,
      aside_input_dim, aside_input_dim, aside_input_dim, aside_input_dim
      };

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
    std::cout << ("context_->allInputDimensionsSpecified() error") << std::endl;
    assert(0);
  }

  for (size_t i = 0; i < device_bindings_.size(); i++) {
    s_device_bindings_.push_back(device_bindings_[i]);
  }

  // warmup
  CUDA_CHECK(cudaMemcpyAsync(d_buffer_, h_buffer_, whole_bytes_ - align_aside_input_bytes_,
                             cudaMemcpyHostToDevice, cuda_stream_));
  cudaStreamSynchronize(cuda_stream_);
}

template<class T>
void _fill(T* ptr, int size, T v) {
  for (int i = 0; i < size; i++) ptr[i] = v;
}

int TrtContext::CaptureCudaGraph() {
  if (graph_created_) return 1;

  // fill test inputs
  auto input_size = max_batch_ * max_seq_len_;
  _fill((int*)host_bindings_[0], input_size, 1);
  _fill((int*)host_bindings_[1], input_size, 1);
  _fill((int*)host_bindings_[2], input_size, 1);
  _fill((float*)host_bindings_[3], input_size, 1.0f);

  _fill((int*)host_bindings_[4], max_batch_, 1);
  _fill((int*)host_bindings_[5], max_batch_, 1);
  _fill((int*)host_bindings_[6], max_batch_, 1);
  _fill((int*)host_bindings_[7], max_batch_, 1);
  _fill((int*)host_bindings_[8], max_batch_, 1);
  _fill((int*)host_bindings_[9], max_batch_, 1);
  _fill((int*)host_bindings_[10], max_batch_, 1);
  _fill((int*)host_bindings_[11], max_batch_, 1);

  CUDA_CHECK(cudaMemcpyAsync(d_buffer_, h_buffer_, whole_bytes_ - align_aside_input_bytes_,
                             cudaMemcpyHostToDevice, cuda_stream_));

  // warm up and let mContext do cublas initialization
  auto status = context_->enqueueV2((void**)device_bindings_.data(), cuda_stream_, nullptr);
  if (!status) {
    cerr << "Enqueue failed\n";
    exit(-1);
  }

  CUDA_CHECK(cudaStreamBeginCapture(cuda_stream_, cudaStreamCaptureModeRelaxed));

  status = context_->enqueueV2((void**)device_bindings_.data(), cuda_stream_, nullptr);
  if (!status) {
    cerr << "Enqueue failed\n";
    exit(-1);
  }

  CUDA_CHECK(cudaStreamEndCapture(cuda_stream_, &graph_));
  CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
  CUDA_CHECK(cudaGraphInstantiate(&instance_, graph_, NULL, NULL, 0));

  CUDA_CHECK(cudaMemcpyAsync(host_bindings_[12], device_bindings_[12], align_aside_input_bytes_,
                             cudaMemcpyDeviceToHost, cuda_stream_));

  graph_created_ = true;

  cout << "profile_idx=" << profile_idx_ << ", CaptureCudaGraph Done!" << endl;

  return 0;
}

int TrtContext::Forward(sample &s) {
  cudaSetDevice(dev_id_);
  int idx = 0;

  auto batch = s.shape_info_0[0];
  auto seq_len = s.shape_info_0[1];
  auto input_bytes = batch * seq_len * sizeof(int);
  auto aside_input_bytes = batch * sizeof(int);
  memcpy(host_bindings_[0], s.i0.data(), input_bytes);
  memcpy(host_bindings_[1], s.i1.data(), input_bytes);
  memcpy(host_bindings_[2], s.i2.data(), input_bytes);
  memcpy(host_bindings_[3], s.i3.data(), input_bytes);

  memcpy(host_bindings_[4], s.i4.data(), aside_input_bytes);
  memcpy(host_bindings_[5], s.i5.data(), aside_input_bytes);
  memcpy(host_bindings_[6], s.i6.data(), aside_input_bytes);
  memcpy(host_bindings_[7], s.i7.data(), aside_input_bytes);
  memcpy(host_bindings_[8], s.i8.data(), aside_input_bytes);
  memcpy(host_bindings_[9], s.i9.data(), aside_input_bytes);
  memcpy(host_bindings_[10], s.i10.data(), aside_input_bytes);
  memcpy(host_bindings_[11], s.i11.data(), aside_input_bytes);

  cudaEvent_t start, stop;
  float elapsed_time = 0.0;

  //cudaStreamSynchronize(cuda_stream_);
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);
  //cudaEventRecord(start, 0);

  CUDA_CHECK(cudaMemcpyAsync(d_buffer_, h_buffer_, whole_bytes_ - align_aside_input_bytes_,
                             cudaMemcpyHostToDevice, cuda_stream_));

  vector<int> v_data(128);
  cudaMemcpyAsync(v_data.data(), device_bindings_[13], 128 * sizeof(int),
                  cudaMemcpyDeviceToHost, cuda_stream_);
  cudaStreamSynchronize(cuda_stream_);

  if (graph_created_) {
    CUDA_CHECK(cudaGraphLaunch(instance_, cuda_stream_));
  } else {

    // printf("before enqueue\n");
    auto status = context_->enqueueV2((void**)device_bindings_.data(), cuda_stream_, nullptr);
    if (!status) {
      cerr << "Enqueue failed\n";
      exit(-1);
    }

  }

  s.out_data.resize(batch);
  //memcpy(s.out_data.data(), host_bindings_[12], batch * sizeof(float));
  CUDA_CHECK(cudaMemcpyAsync(s.out_data.data(), device_bindings_[start_binding_idx_ + 12], batch * sizeof(float),
                             cudaMemcpyDeviceToHost, cuda_stream_));
  cudaStreamSynchronize(cuda_stream_);

  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&elapsed_time, start, stop);

  //cout << "batch=" << max_batch_ << ", seq_len=" << max_seq_len_
       //<< ", enqueue time=" << elapsed_time << "ms" << endl;
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
