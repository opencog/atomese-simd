OpenCL Examples
===============

One example:
* `atomese-kernel.scm` demonstrates how to use Atomese to open a channel
  to an OpenCL device, and then send a compute kernel and some floating-
  point vector data to it. The results from the computation are pulled
  back into the AtomSpace.  The data is operated on by the kernels
  defined in `vec-kernel.cl`.
