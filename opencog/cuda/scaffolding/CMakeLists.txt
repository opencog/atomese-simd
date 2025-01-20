
# sniff test
ADD_EXECUTABLE(cuda-hello cuda-hello.cu)
SET_TARGET_PROPERTIES(cuda-hello PROPERTIES CUDA_ARCHITECTURES native)
