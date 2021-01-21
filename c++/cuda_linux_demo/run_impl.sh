mkdir -p build
cd build
rm -rf *

DEMO_NAME=model_test

WITH_MKL=ON
WITH_GPU=ON
USE_TENSORRT=OFF

LIB_DIR=/home/heya02/dist_test/serving_0.4.1/0118_code/Serving/build/third_party/install/Paddle/
CUDNN_LIB=/usr/lib64/
CUDA_LIB=/usr/local/cuda/lib64/
TENSORRT_ROOT=/path/to/trt/root/dir

cmake .. -DPADDLE_LIB=${LIB_DIR} \
  -DWITH_MKL=${WITH_MKL} \
  -DDEMO_NAME=${DEMO_NAME} \
  -DWITH_GPU=${WITH_GPU} \
  -DWITH_STATIC_LIB=OFF \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DCUDNN_LIB=${CUDNN_LIB} \
  -DCUDA_LIB=${CUDA_LIB} \
  -DTENSORRT_ROOT=${TENSORRT_ROOT}

make -j
