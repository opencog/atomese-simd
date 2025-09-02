
# Auto-generated types files are in the build directory.
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

ADD_LIBRARY (opencl SHARED
	OpenclFloatVector.cc
	OpenclStream.cc
)

# Without this, parallel make will race and crap up the generated files.
ADD_DEPENDENCIES(opencl opencl_atom_types)

TARGET_LINK_LIBRARIES(opencl
	opencl-types
	${SENSORY_LIBRARIES}
	${ATOMSPACE_LIBRARIES}
	OpenCL::OpenCL
)

INSTALL (TARGETS opencl EXPORT AtomSpaceTargets
	DESTINATION "lib${LIB_DIR_SUFFIX}/opencog"
)

INSTALL (FILES
	OpenclFloatVector.h
	OpenclStream.h
	DESTINATION "include/opencog/atoms/opencl/"
)
