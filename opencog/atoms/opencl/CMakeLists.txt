
# Auto-generated types files are in the build directory.
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

ADD_LIBRARY (opencl-atoms SHARED
	OpenclFloatValue.cc
	OpenclNode.cc
	OpenclValue.cc
)

# Without this, parallel make will race and crap up the generated files.
ADD_DEPENDENCIES(opencl-atoms opencl_atom_types)

TARGET_LINK_LIBRARIES(opencl-atoms
	opencl-types
	${SENSORY_LIBRARIES}
	${ATOMSPACE_LIBRARIES}
	OpenCL::OpenCL
)

INSTALL (TARGETS opencl-atoms EXPORT AtomSpaceTargets
	DESTINATION "lib${LIB_DIR_SUFFIX}/opencog"
)

INSTALL (FILES
	opencl-headers.h
	OpenclFloatValue.h
	OpenclNode.h
	OpenclValue.h
	DESTINATION "include/opencog/atoms/opencl/"
)
