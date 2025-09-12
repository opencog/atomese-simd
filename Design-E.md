Design Notes E
==============
September 2025

Notes about wiring, composability and flow.

Status: Basic GPU operations seem to work and seem to have a reasonable
API. Time to review higher order goals. This is a recap of unimplemented
ideas from Design Notes C.

There are various unfulfilled design goals:

* There should be a way of creating wiring diagrams, along which data
  flows. Abstractly, this would be sensory data, flowing in from some
  sensor, being transformed via sequences of GPU kernels, and then
  updating AtomSpace memory and driving motors.

* Allow basic DL/NN ideas like transformers to be written in Atomese.
  That is, there should be a way of converting DL/NN pseudocode from
  published papers into Atomese, and then run it.

* Implement an Atomese version of LTN, Logic Tensor Networks, but
  generalizing it so that any kind of logic can be encoded, and not
  just the "real logic" of LTN. Specifically, want to be able to encode
  assorted modal logics, including epistemic logic.

* Provide semantic elements. The current demo includes two kernels:
  one that adds a pair of vectors, and another that multiplies them.
  These are representations of the abstraction provided by `PlusLink`
  and `TimesLink`. It seems appropriate that these two kernels should
  be stored in or associated with `PlusLink` and `TimesLink`, so that
  the Atomese is written with these, and not with the raw kernels.

* Provide composable, compilable elements. A large subset of Atomese
  allows for the writing of abstract syntax trees, representing
  processing flows. If such trees include both  `PlusLink` and
  `TimesLink` and these are composed together, then the data should
  never leave the GPU, but should be processed in-place. That is, the
  OpenCL kernels themselves should be composed and compiled "on the
  fly", to perform the desired operation.

It seems like all of these should be possible. The groundwork for some
of this has been laid, and so some of these can be tackled. Perhaps
implementing a library for the core AtomSpace arithmetic functions is
a reasonable next step: so, having `PlusLink` and `TimesLink run on the
GPU.

----
