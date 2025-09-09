Design Notes C
==============
September 2025

Notes about wiring, composability and flow.

Status: The most basic demo of adding and multiplying vectors works.
Nothing flows in this demo: it's static in it's design. One specifies
a collection of inputs and outputs, using a `Section`, ship the inputs
to the GPU, and wait for results.

There are various unfulfilled design goals:

* There should be a way of creating wiring diagrams, along which data
  flows. Abstractly, this would be sensory data, flowing in from some
  sensor, being transformed via sequences of GPU kernels, and then
  updating AtomSpace memory and driving motors.

* Allow basic DL/NN ideas like transformers to be written in Atomese.
  That is, there should be a way of converting DL/NN pseudocode from
  published papers into Atomese, and then run it.

* Implement an Atomese version of LTN, Logic Tensor Newtorks, but
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

It seems like all of these should be possible, but we're not there yet.
Time to review these ideas in greater depth, to see what design could
emerge.

Flow Demos
----------
The current sensory demos use a `FilterLink`/`RuleLink` combo to
represent a processing element. The mape from this to the SIMD Atomese
is unclear.

There's a recurrng confusion in my thinking: confusing `Sections`, which
represent general jigsaws, and `RuleLink`s, which are specific jigsaws
that are natural processing elements for flows. They are similar, but
not the same, and the similarity is the source of the confusion.  The
jigsaws desribed by `Sections` are primarily declarative, and they
describe the connectors on jigsaws, and thus describe how they fit
together. The `RuleLink`s feel like a special case, except that they
use `VariableList`s to describe the conectors, instead of using
`Section`s. There's an unresolved tension between these two. It hampers
clear design.

Let's review the `RuleLink`. It has the form:
```
    (RuleLink
        (VariableList (Variable "$x") (Variable "$y") ...)
        (Signature ... ) ; Item pattern to recognize
        (Signature ... ) ; Item to generate.
```
What is the corresponding `Section` for this? I don't know. There are
several choices. One is fairly trivial: the jigsaw that has one input (a
stream of items) and one output (another stream of items). If the stream
does not contain any items that match the rule recognizer pattern, that
item is discarded from the stream.

GPU State
---------
In the foreground is the problem of the "correct" representation of
vectors accessible to the GPU.

This is the accumulator issue.  Several solutions what is best?

----
