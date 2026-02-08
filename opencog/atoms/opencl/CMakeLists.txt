
# Auto-generated types files are in the build directory.
INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR})

ADD_LIBRARY (opencl-atoms SHARED
	GenIDL.cc
	OpenclDataValue.cc
	OpenclFloatValue.cc
	OpenclJobValue.cc
	OpenclNode.cc
	OpenclNode-cache.cc
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
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/opencog"
)

INSTALL (FILES
	opencl-headers.h
	OpenclDataValue.h
	OpenclFloatValue.h
	OpenclJobValue.h
	OpenclNode.h
	DESTINATION "include/opencog/atoms/opencl/"
)
