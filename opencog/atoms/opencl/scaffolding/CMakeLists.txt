
INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIRS})

# Build scaffolding tool
ADD_EXECUTABLE(show-ocl-hw show-ocl-hw.cc scaffolding.cc)
ADD_EXECUTABLE(run-hello-world run-hello-world.cc scaffolding.cc)
ADD_EXECUTABLE(run-vec-mult run-vec-mult.cc scaffolding.cc)
ADD_EXECUTABLE(run-flow-vec run-flow-vec.cc scaffolding.cc)

# TARGET_LINK_LIBRARIES(show-ocl-hw ${OPENCL_LIBRARIES})
TARGET_LINK_LIBRARIES(show-ocl-hw OpenCL::OpenCL)
TARGET_LINK_LIBRARIES(run-hello-world OpenCL::OpenCL)
TARGET_LINK_LIBRARIES(run-vec-mult OpenCL::OpenCL)
TARGET_LINK_LIBRARIES(run-flow-vec OpenCL::OpenCL)

# Perform an "offline build" of the demo sources.
# Do this only if clang and llvm-spirv are installed.
# This is optional for the demos. Requires OpenCL version 2.0
# See https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/os_tooling.md
# for detailed documentation.
IF (HAVE_CLANG AND HAVE_LLVM_SPIRV)
	ADD_CUSTOM_COMMAND(
		OUTPUT vec-mult.bc
		COMMAND ${CLANG}
		ARGS -c ${CMAKE_CURRENT_SOURCE_DIR}/vec-mult.cl -target spir -O0 -emit-llvm -o vec-mult.bc
		DEPENDS vec-mult.cl)

	ADD_CUSTOM_COMMAND(
		OUTPUT vec-mult.spv
		COMMAND ${LLVM_SPIRV}
		ARGS ${CMAKE_CURRENT_BINARY_DIR}/vec-mult.bc -o vec-mult.spv
		DEPENDS vec-mult.bc)

	ADD_CUSTOM_TARGET(vec_mult_lib DEPENDS vec-mult.spv)
	ADD_DEPENDENCIES(run-flow-vec vec_mult_lib)
ENDIF (HAVE_CLANG AND HAVE_LLVM_SPIRV)
