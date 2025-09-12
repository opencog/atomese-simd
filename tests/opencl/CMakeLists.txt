
INCLUDE(AddCxxtest)

# Unit test for GenIDL
ADD_CXXTEST(GenIDLUTest)
TARGET_LINK_LIBRARIES(GenIDLUTest
	opencl-atoms
	${ATOMSPACE_LIBRARIES}
)

ADD_GUILE_TEST(OpenclVecTest opencl-vec-test.scm)

SET_PROPERTY(TEST OpenclVecTest
	PROPERTY DEPENDS opencl-atoms opencl.scm)
