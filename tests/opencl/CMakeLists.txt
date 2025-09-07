
ADD_GUILE_TEST(OpenclVecTest opencl-vec-test.scm)

SET_PROPERTY(TEST OpenclVecTest
	PROPERTY DEPENDS opencl-atoms opencl.scm)
