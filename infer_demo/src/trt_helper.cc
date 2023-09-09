#include "trt_helper.h"

#include <string>
#include <fstream>
#include <sstream>

using namespace std;

// BEGIN_LIB_NAMESPACE {

cuda_shared_ptr<void> CpuToDevice(const std::vector<int>& shape, __int64_t* data_ptr) {
  void *d_ptr;
  auto cpu_ptr = static_cast<void *>(data_ptr);
  int data_size = 1;
  for (int i = 0; i < shape.size(); i++) data_size *= shape[i];
  auto ret = cudaMalloc(&d_ptr, data_size * sizeof(__int64_t));
  printf("int memory\n");
  if (ret) printf("memory error\n");
  ret = cudaMemcpy(d_ptr, cpu_ptr, data_size * sizeof(__int64_t), cudaMemcpyHostToDevice);
  if (ret) printf("memory error\n");
  cuda_shared_ptr<void> cuda_ptr;
  make_cuda_shared(cuda_ptr, d_ptr);
  return cuda_ptr;
}

cuda_shared_ptr<void> CpuToDevice(const std::vector<int>& shape, float* data_ptr) {
  void *d_ptr;
  auto cpu_ptr = static_cast<void *>(data_ptr);
  int data_size = 1;
  for (int i = 0; i < shape.size(); i++) data_size *= shape[i];
  auto ret = cudaMalloc(&d_ptr, data_size * sizeof(float));
  printf("float memory\n");
  if (ret) printf("memory error\n");
  ret = cudaMemcpy(d_ptr, cpu_ptr, data_size * sizeof(float), cudaMemcpyHostToDevice);
  if (ret) printf("memory error\n");
  cuda_shared_ptr<void> cuda_ptr;
  make_cuda_shared(cuda_ptr, d_ptr);
  return cuda_ptr;
}

void DeviceToCpu(const std::vector<int>& shape, cuda_shared_ptr<void> cuda_ptr, float* data_ptr) {
  int data_size = 1;
  for (int i = 0; i < shape.size(); i++) data_size *= shape[i];
  if (data_size == 0) {
    std::cout << "data_size == 0" << std::endl;
    assert(0);
  }
  auto d_ptr = static_cast<void *>(data_ptr);
  auto ret = cudaMemcpy(d_ptr, cuda_ptr.get(), data_size * sizeof(float), cudaMemcpyDeviceToHost);
  printf("copy back\n");
  if (ret) printf("memory error\n");
}

TrtLogger::TrtLogger(nvinfer1::ILogger::Severity level) : level_(level) {
  /*
  int argc = 1;
  const char* argv[] = {"Forward"};
  START_EASYLOGGINGPP(argc, argv);
  // Load configuration from file
  el::Configurations conf("forward_log.conf");
  // Reconfigure single logger
  el::Loggers::reconfigureLogger("default", conf);
  // Actually reconfigure all loggers instead
  el::Loggers::reconfigureAllLoggers(conf);
  */
}

nvinfer1::ILogger& TrtLogger::getTRTLogger() { return *this; }

// trt logger
void TrtLogger::log(Severity severity, const char* msg) noexcept {
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


TrtHepler::TrtHepler(std::string model_param, int dev_id)
    : _dev_id(dev_id), _model_param(model_param) {
  { // read model, deserializeCudaEngine and createExecutionContext
    std::ifstream t(_model_param);  // string pth
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string contents(buffer.str());

    CUDA_CHECK(cudaSetDevice(_dev_id));
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));

    TrtLogger trt_logger;
    auto runtime = MakeUnique(nvinfer1::createInferRuntime(trt_logger.getTRTLogger()));
    auto e = runtime->deserializeCudaEngine((void*)contents.c_str(),
                                            contents.size(), nullptr);
    engine_ = MakeShared(e);
    context_ = MakeShared(engine_->createExecutionContext());
    context_->setOptimizationProfile(0);
  }

  // // copy max input and output size from node
  // hd_inputs_.resize(input_num);
  // hd_outputs_.resize(output_num);

  // device_bindings_.resize(input_num + output_num);

  // inputs_dims_.resize(input_num);
}

// int TrtHepler::Forward(std::vector<std::shared_ptr<HostDeviceMat>>& inputs,
                       // std::shared_ptr<HostDeviceEntireAllocator> hd_input_allocator_ptr,
                       // std::vector<std::shared_ptr<HostDeviceMat>>& outputs,
                       // std::shared_ptr<HostDeviceEntireAllocator> hd_output_allocator_ptr) {

int TrtHepler::Forward(sample& s) {
  cudaSetDevice(_dev_id);
  auto rc_ids_tensor = CpuToDevice(s.shape_info_0, s.i0.data());
  auto sent_ids_tensor = CpuToDevice(s.shape_info_1, s.i1.data());
  auto pos_ids_tensor = CpuToDevice(s.shape_info_2, s.i2.data());
  std::vector<__int64_t> mask_idx;
  for (int i = 0; i < s.shape_info_3[0]; i++) {
    int mask_len = 0;
    for (int j = 0; j < s.shape_info_3[1]; j++) {
      if (s.i3[j] != 0.0) {
        mask_len++;
      } else break;
    }
    mask_idx.push_back(mask_len);
  }
  vector<int> s3_shape = {s.shape_info_3[0], 1};
  std::cout << s3_shape[0] << std::endl;
  auto input_mask_tensor = CpuToDevice(s3_shape, mask_idx.data());  //modify: 1 * batch
  auto tmp6_tensor = CpuToDevice(s.shape_info_4, s.i4.data());
  auto tmp7_tensor = CpuToDevice(s.shape_info_5, s.i5.data());
  auto tmp8_tensor = CpuToDevice(s.shape_info_6, s.i6.data());
  auto tmp9_tensor = CpuToDevice(s.shape_info_7, s.i7.data());
  auto tmp10_tensor = CpuToDevice(s.shape_info_8, s.i8.data());
  auto tmp11_tensor = CpuToDevice(s.shape_info_9, s.i9.data());
  auto tmp12_tensor = CpuToDevice(s.shape_info_10, s.i10.data());
  auto tmp13_tensor = CpuToDevice(s.shape_info_11, s.i11.data());

  void* out_ptr;
  auto ret_ = cudaMalloc(&out_ptr, s.shape_info_0[0] * sizeof(float));  // -1 * 1
  cuda_shared_ptr<void> cuda_out_ptr;
  make_cuda_shared(cuda_out_ptr, out_ptr);
  // auto inputs = task_node->hd_batch_inputs_ptr_;
  // auto outputs = task_node->hd_batch_outputs_ptr_;
  // auto hd_input_allocator_ptr = task_node->hd_input_allocator_ptr_;
  // auto hd_output_allocator_ptr = task_node->hd_output_allocator_ptr_;

  // const char* env_p = std::getenv("TRTDIY_LOG");
  const char* env_p = "1";
  cudaEvent_t start, stop;
  float elapsed_time = 0.0;

  // size_t input_num = inputs.size();
  // size_t output_num = outputs.size();

  int binding_idx = 0;
  std::vector<std::vector<int>> input_dims = {s.shape_info_0, s.shape_info_1, s.shape_info_2, s3_shape,
                                              s.shape_info_4, s.shape_info_5, s.shape_info_6, s.shape_info_7,
                                              s.shape_info_8, s.shape_info_9, s.shape_info_10, s.shape_info_11};
  // set device_bindings_ and setBindingDimensions
  for (size_t i = 0; i < input_dims.size(); i++) {
    std::vector<int> dims_vec = input_dims[i];
    nvinfer1::Dims trt_dims;
    trt_dims.nbDims = static_cast<int>(dims_vec.size());
    memcpy(trt_dims.d, dims_vec.data(), sizeof(int) * trt_dims.nbDims);
    context_->setBindingDimensions(binding_idx, trt_dims);
    binding_idx ++;
  }

  // getBindingDimensions, resize outputs and set device_bindings_
  // for (size_t i = 0; i < output_num; i++) {
  //   auto trt_dims = context_->getBindingDimensions(binding_idx);
  //   DimsVector dims_vec(trt_dims.nbDims);
  //   memcpy(dims_vec.data(), trt_dims.d, sizeof(int) * trt_dims.nbDims);

  //   outputs[i]->Resize(dims_vec);

  //   device_bindings_[binding_idx] = outputs[i]->GetDeviceData();
  //   binding_idx ++;
  // }

  if (!context_->allInputDimensionsSpecified()) {
    //gLogFatal << "context_->allInputDimensionsSpecified() error";
    std::cout << ("context_->allInputDimensionsSpecified() error") << std::endl;
    assert(0);
  }

  // cudaStreamSynchronize(cuda_stream_);
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);

  // set the input dim

  void *device_bindings[13] = {rc_ids_tensor.get(), sent_ids_tensor.get(), pos_ids_tensor.get(),
                               input_mask_tensor.get(), tmp6_tensor.get(), tmp7_tensor.get(),
                               tmp8_tensor.get(), tmp9_tensor.get(), tmp10_tensor.get(),
                               tmp11_tensor.get(), tmp12_tensor.get(), tmp13_tensor.get(),
                               cuda_out_ptr.get()};
  printf("before enqueue\n");
  bool ret = context_->enqueueV2(device_bindings, cuda_stream_, nullptr);
  if (!ret) {
    //gLogError << " context_->enqueueV2 failed!";
    std::cout << ("context_->enqueueV2 failed!") << std::endl;
    return -100;
  }
  // cudaStreamSynchronize(cuda_stream_);
  // cudaEventRecord(stop, 0);
  // cudaEventSynchronize(stop);
  // cudaEventElapsedTime(&elapsed_time, start, stop);
  // // auto dim_str = Dims2String(_inputs_dims[0]);
  // LOG(INFO) << "input batch=" << task_node->req_num_ << ", enqueue time=" << elapsed_time << "ms";
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);

  // cudaStreamSynchronize(cuda_stream_);
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);
  // cudaEventRecord(start, 0);

  cudaMemcpy(s.out_data.data(), cuda_out_ptr.get(), s.shape_info_0[0] * sizeof(float), cudaMemcpyDeviceToHost);
  cudaStreamSynchronize(cuda_stream_);
  struct timeval tv;
  gettimeofday(&tv, NULL);
  s.timestamp = tv.tv_sec * 1000000 + tv.tv_usec;
  // if (env_p != nullptr) {
    // cudaStreamSynchronize(cuda_stream_);
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsed_time, start, stop);
    // // auto dim_str = Dims2String(_inputs_dims[0]);
    // LOG(INFO) << "input batch=" << task_node->req_num_ << ", CopyDeviceToHostAsync time=" << elapsed_time << "ms";
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
  // }

}

TrtHepler::~TrtHepler() {
  CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
}

// } // BEGIN_LIB_NAMESPACE

