OpenCL Examples
===============

Two example:
* `atomese-kernel.scm` demonstrates how to use Atomese to open a channel
  to an OpenCL device, and then send a compute kernel and some floating-
  point vector data to it. The results from the computation are pulled
  back into the AtomSpace.  The data is operated on by the kernels
  defined in `vec-kernel.cl`.

* `accumulator-bad.scm` demonstrates how to define a location in the
  AtomSpace which can be used to hold vector data. The demo is "bad"
  only in that it performs excessive data movement. But it works, and
  the point is to show how to move data.
