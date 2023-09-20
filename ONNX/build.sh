#python ONNX/onnx2trt.py -m=../sti2_data/model/onnx_infer_model/model.onnx -o=trt_models/ernie_onnx_fp32.engine 
python ONNX/onnx2trt.py -f -m=../sti2_data/model/onnx_infer_model/model.onnx -o=trt_models/ernie_onnx_fp16.engine 
