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

Rewriting
---------
Consider the task of mapping the Atomese `PlusLink` so that, when
invoked/executed in the right way, it runs on the GPU. There are several
ways to implement this. The knee-jerk, brute-force but stupid way is to
write a bunch of C++ code that intercepts the attempt to execute
`PlusLink`, dependent on the context, and re-route it to the GPU.
Of course this can be made to work; its brute-forcing the solution.

The other way is to treat it as a re-write. That is, create a `RuleLink`
that accepts (recognizes) `(Plus a b)` and turns it into
```
   (Section (Item "vec_add")
      (FloatValue rslt) (FloatValue a) (FloatValue b))
----
Well, not that literally, because we'd have to wire `rslt` into
something; its more complicated. Also `Plus` is unbounded, so it would
need to be some arithmetic fold. But you get the idea...

So this becomes a graph-rewrite problem: Accept as input functional
expressions involving `PlusLink` etc and rewrite thehm into Opencl
Atomese. Once rewritten, they can be executed.

Perhaps easier to map are `LambdaLink` because they have explicit
`VariableList` declarations, with each being a `TypedVariable`, and so
this maps more smoothly and easily to the OpenCL C/C++ style interfaces.

GenIDL
------
The above thoughts also suggest that the implementation of GenIDL is
wrong. The original vision (described in [Design-C](./Design-C.md)) was
to convert OpenCL function prototype declarations like
```
    kernel void vec_mult(global double *prod,
                         global const double *a,
                         global const double *b,
                         const unsigned long sz)
```
into Atomese
```
    (Section
        (ItemNode "vec_mult")
        (ConnectorSeq
            (Connector (Type 'FloatValue) (Sex "output"))
            (Connector (Type 'FloatValue) (Sex "input"))
            (Connector (Type 'FloatValue) (Sex "input"))
            (Connector (Type 'FloatValue) (Sex "size"))))
```
And that's what it does.  However, it might be better to convert the
OpenCL declaration into Atomese that is much much more faithful to
it's original form, and then rewrite *that* (using `RuleLink`) into
the atomese-simd interfaces.

Malleability
------------
What's the point of doing this? Its to
