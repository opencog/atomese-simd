OpenCL Examples
===============

Two example:
* `atomese-kernel.scm` demonstrates how to use Atomese to open a channel
  to an OpenCL device, and then send a compute kernel and some floating-
  point vector data to it. The results from the computation are pulled
  back into the AtomSpace.  The data is operated on by the kernels
  defined in `vec-kernel.cl`.

* `accumulator.scm` demonstrates how to define a location in the
  AtomSpace which shadows a vector held on the GPU. That is, the vector
  "lives" on the GPU, and is downloaded back to to the system when
  it is "examined", e.g. to be printed to stdout.
