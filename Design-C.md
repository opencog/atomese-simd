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
represent a processing element. The map from this to the SIMD Atomese
is unclear.

There's a recurrng confusion in my thinking: confusing `Sections`, which
represent general jigsaws, and `RuleLink`s, which are specific jigsaws
that are natural processing elements for flows. They are similar, but
not the same, and the similarity is the source of the confusion.  The
jigsaws described by `Sections` are primarily declarative, and they
describe the connectors on jigsaws, and thus describe how they fit
together. The `RuleLink`s feel like a special case, except that they
use `VariableList`s to describe the connectors, instead of using
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
One problem that arises is the proper design of the accumulator.
Conceptually, there are two vectors: A, which is initially zero, and
generally lives on the GPU "permanently", and is infrequently examined,
and a sequence for vectors B, which are added to A. The vectors B might
be coming from two different locations: they might be getting uploaded
to to GPU from system memory, or they might be generated on the GPU.

To implement this, we need to model the vector A as a "thing" in the
"external world" (the GPU) which has some constancy of existence, but
whose content changes. Using a compiler metaphor, it is a storage
location whose content changes; a register or a moemory location.

The `OpenclFloatValue` provides a mechanism to sample from this
location. The C++ FloatValue can be sampled repeatedly, but this would
require attaching the `update()` method to an Opencl device, context
and event queue, which is not practical in the current design. The C++
FloatValue is also not directly updatedable; the Value interfaces are,
in general, not updatedable; one must create a new Value. That design
was chosen to get thread safety. It should not be changed.

To get an updateable `OpenclFloatValue` vecttor that can be both read,
repeatedly, and changed, repeatedly, requires use of the `OpenclNode`
`*-read-*` and `*-write-*` methods. These seem adequate for the task.
Currently, `OpenclNode` only accepts kernels, and not the vectors
themselves. That is easily fixed.

Reads seem easy enough to deal with: the `OpenclFloatValue` holds the
`cl::Buffer` needed for external ref constancy. That is, each
`cl::Buffer` is a handle to that "thing" in the "external world".
Reading from this is no problem, as long as that reference is retained.
Writing is a problem, because 'FloatValue' has no generic 'set' method,
so we need some way of updating contents without losing the `cl::Buffer`
handle. This could be done with a custom private/proected API that is
accessible only to `OpenclNode`.

An alternative design would be to have an `OpenclNumberNode`, which
provides AtomSpace constancy. But this would need to be given some
abstract name, since the numerical value would be changing. This, not
a `NumberNode` after all, but a `OpenclVectorNode` which can be given a
specific name. It could then manage reads and write ... except it can't
do this without a connext and a device; and since `OpenclNode` already
has this, then may as well have `OpenclNode` do that management. So
no new Node is needed.

The OpenclKernelNode
--------------------
An `OpenclKernelLink` is needed to manage the specific kernel that is to
be run. The current API is muddled: we need to be able to declare the
following:
 * "Here's a vector in RAM; the kernel needs read access to it."
   (Who is responsible for uploading it? SVM impies that explicit
   upload is not needed!?)
 * "Here's a vector you already have, the kernel will update it."
   (Does not imply a download, or an upload; so perhaps it's already
   available to the GPU and needs no management from us.)

An alternative design point would be to have an `OpenclKernelLinkValue`
so that `FloatValues` can be stuffed into it directly, instead of using
`ValueOf` references.  But the long-term design is to flow, and so ...
Hmm.

An alternative design is to have an `OpenclKernelNode`. This makes
sense, at it is the specific kernel name that is being invoked that is
important.

Great! This brings up back to the original `RuleLink` vs. `Section` flow
description. Each specific `OpenclKernelNode` needs to have an
adjoinging declaration of what it's valid inputs and outputs are.
We have two choices for providing this description. The old-fashioned,
traditional description would be to have a `VariableList` of
`TypedVariable` indicating what it's inputs are. The problem here is
that there is no particular way of describing the output.

The `RuleLink` was sort-of imagined to describe inputs and outputs, but
has been co-opted by the `FilterLink`. The use of variables in the
`RuleLink` means that it is explicitly a re-write rule, in that it
tracks variables in both the input and the output. By contrast, a kernel
operation is not a rewrite: the outputs depend on the inputs, but they
are not monotonic functions of the input variables.

That leaves `Section` as the only viable candidate for describing inputs
and outputs. Excellent: we finally arrive at an actual need for chaining
and checking the chains! OK, so what should the connectors look like?
Link Grammar style connectors would look like this:
```
    (Connector
        (Type 'FloatValue)
        (Sex "input"))
```
This is a paring of a traditional type declaration, together with a
direction (sex). The type declaration can be complicated, in principle:
```
    (Connector
        (SignatureLink ...)
        (Sex "input"))
```
with the usual richenesss of Signatures allowed.

The current `vec_add` kernel in the demos then has the following form:
```
    (Section
        (OpenclKernelNode "vec_add")
        (ConnectorSeq
            (Connector (Type 'FloatValue) (Sex "output"))
            (Connector (Type 'FloatValue) (Sex "input"))
            (Connector (Type 'FloatValue) (Sex "input"))
            (Connector (Type 'FloatValue) (Sex "size"))))
```
Note that the order of the connectors in the `ConnectorSeq` must match
the c/c++ code in the program.

The `size` connector is interesting. In pricniple, it could be implicit,
guessed from the sizes of the vectors. In practice, it seems to be a
required part of the kernel API: the kernel needs to be told what the
length of the vectors are, explicitly so.

Where is this API description kept? Well, the `*-description-*` message
sent to the `(OpenclKernelNode "vec_add")` node should return this.
I guess this is hand-coded, for now.

Kernel Execution
----------------
What happens when
```
    (cog-execute! (OpenclNode ...) (Predicate "*-write-*")
        ( ... data ...))
```
is sent? Well, the `*-write-*` method has to validate that the data
is actually of the format defined in the `Section` declaration. If not,
it throws an error. (We throw, instead of silently failing, to ease
debuggability.)

Here's the logic for the (implicit) type conversions in the data stream:

 * If an input vector is a `NumberNode` or a `FloatValue`, a
   corresponding `OpenclFloatValue` is created, copying the vector
   out of the `NumberNode` or `FloatValue`, and marking it as
   `CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR`.

 * If an input vector is already an `OpenclFloatValue`, nothing needs
   to be done, except maybe to or-in `CL_MEM_COPY_HOST_PTR` ??

 * If an output vector is already an `OpenclFloatValue`, nothing needs
   to be done, except maybe to or-in `CL_MEM_READ_WRITE`.

The only problem I see here is that these `CL_MEM_*` flags are set up
by the `cl::Buffer()` ctor; can they be changed later? Hmm. Well,
certainly `enqueueReadBuffer` can specify the `void*` data pointer at a
later time. Also `enqueueWriteBuffer` allows the data location to be
deferred, and so `CL_MEM_COPY_HOST_PTR` does not need to be specified at
ctor time.

A `cl::Buffer` created read-only cannot be converted to read-write.

The current demo allows the output ector to be specified as a
`(TypeNode 'FloatVector)`, after which it implicitly creates the
required `OpenclFloatValue` and sets it to the correct size, and then
returns it with the `*-read-*` method. This is OK, I guess, unless there
are two outputs, in which case `*-read-*` needs to return a `LinkValue`
holding both.

Which raises an old issue: due to the async nature, the `*-read-*`
should return not just the outputs, but the entire wrapped kernel.
A later read can retrieve individual outputs, as required.

OK, I think that's a plan. Lets do it.

----
