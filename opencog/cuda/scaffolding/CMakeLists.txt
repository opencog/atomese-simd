
# sniff test
ADD_EXECUTABLE(cuda-hello cuda-hello.cu)

# Supported architectures can be listed with
#   nvidia-smi --query-gpu=compute_cap --format=csv
SET_TARGET_PROPERTIES(cuda-hello PROPERTIES CUDA_ARCHITECTURES "50;72;89")
