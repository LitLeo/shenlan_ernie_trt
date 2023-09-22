# fp32 build
#python3 API/builder.py -p ../sti2_data/model/paddle_infer_model -o trt_models/ernie_api_fp32.engine
#python3 API/builder.py -f -p ../sti2_data/model/paddle_infer_model -o trt_models/ernie_api_fp16.engine
python3 API/builder.py -g -p ../sti2_data/model/paddle_infer_model -o trt_models/ernie_api_fp16.engine

