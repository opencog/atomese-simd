Design Notes
============
Notes about OpenCL interfaces and how they impact Atomese design.

* SVM "Shared Virtual Memory" shares vector data between GPU and
  CPU, avoiding explicit copy semantics.  The ctors need `cl::Context`.
  Older hardware doesn't support SVM.

* Non-SVM vector I/O is done with a `cl::Buffer` object (per vector).
  Explicit copyin/copyout is needed for each access.
  The `cl::Buffer()` ctor requires a `cl::Context`.

* The kernel itself needs only a `cl::Program` which holds the
  actual kernel code.

* Kernels are executed by placing them onto a `cl::CommandQueue`.
  This queue takes both a `cl::Context` and also a `cl::Device`
  in it ctor. Kernels are executed async.

* Programs written in `*.cl` or `*.clcpp` can be "offline compiled" and
  placed into an `*.spv` binary file. Requires OpenCL version 2.0.

Design alternatives
-------------------
Different ideas for communicating with GPUs.

* Use `GroundedSchemaNode`s to wrap kernels. Old-style, yucky.
  Why yucky? Because its stateless: besides the string name of the
  node, nothing is stored in the atom. There's no open/close phasing.
  Its just a function call. Maps poorly onto stateful I/O.

* Use `StorageNode`s to wrap kernels. This presents other issues.
  First, traditional `StorageNodes` were meant to send/receive atoms,
  while here, we need to send/receive vectors represented as
  `FloatValue`s.  Maybe other kinds of vectors, but for now,
  `FloatValue`s.  The other problem is that the `StorageNode`
  API sits outside of Atomese proper, and uses instead a collection
  of calls `cog-open` and `cog-close` to manage connection state.
  Which is adequate for the intended use, but leaves something to be
  desired.

* Create something new, inspired by `BackingStore`. This removes some
  of the issues with StorageNodes, but leaves others in place. One
  is that, again, the API sits outside of Atmoese. The other is that
  this becomes yet another engineered solution. Of course, engineered
  solutions are inescapable, but... for example: the traditional unix
  open/close/read/write 4-tuple provides an abstraction that works very
  well for many I/O tasks: open/close to manage the channel, and
  read/write to use it. But the OpenCL interfaces do not map cleanly
  to open/close/read/write. They use a more complex structure.

The abstraction I'm getting is this: open near point, open far point,
select far-point message recipient, send message. That mapping would
have psuedocode like so:
```
   open_near_point() {
      Using cl::Device
      Create cl::Context and retain pointer to it
   }

   open_far_point() {
      Using cl::Context from near point, and externally specified
      cl::Device, create a cl::CommandQueue. Retain pointer to the
      cl::CommandQueue. Its the commo handle.
   }

   select_recipient() {
      Use the externally supplied cl::Program, kernel name string,
      number of inputs, outputs, and size.
      Create kernel, retain pointer to it.
      Call cl::Kernel::setArgs() to wire up the kernel.
   }

   send() {
      Using the selected recipient,
      Call cl::CommandQueue::enqueueWriteBuffer() to send data.
		Wait on cl::Event
   )

   exec() {
      Using the selected recipient,
      Call cl::CommandQueue::enqueueNDRangeKernel() to run kernel.
		Wait on cl::Event
   )

   recv() {
      Using the selected recipient, wait on cl::Event
      Call cl::CommandQueue::enqueueReadBuffer() to get data.
		Wait on cl::Event
   }
```
Caveat: the above is already an oversimplification of the OpenCL
interfaces, because a `cl::Conext` is not created out of thin air, but
requires a vector of `cl::Device` in it's ctor. And devices need
`cl::Platform`. Lets ignore these additional complexities, for the
moment.

The above API is more complex than open/close/read/write. There are
four choices:
* Collapse the first three steps into a generic `open()`.
  Collapse the send and exec steps into a generic `write()`.
* The multi-stage open-near, open-far, select-recipient is fairly
  generic for network communications. e.g. for tcp/ip, open-near
  corresponds to opening local socket (initializing ethernet adapter)
  open-far corresponds to the remote socket, and select-recipient
  corresponds to selecting the port number.
  This multi-stage open-and-contact-recipient can be codified as
  "sufficiently generic", claiming that e.g. CUDA would also fit
  into this model.
* Recognize the above as a cascade of opens(), each requiring the prior
  so that it resembles the peeling back of an onion. That is, provide
  three distinct objects, each having a single `open()` on it, and
  requiring these to be chained, in order to open channel to far point
  station.
* Recognize that the peeling-of-an-onion model is too 'linear', and that
  there is a network of interactions between the various `cl::` classes.
  That network is not a line, but a DAG. Encode the DAG as Atomese.
  That is, create an Atom that is more-or-less in 1-1 correspondence
  with the OpenCL classes. Communication then requires connecting the
  Atomese DAG in the same way that the OpenCL API expects the
  connections.

The first option seems easiest. The fourth option seems most generic.
How would this work, in practice?

Wiring Diagrams
---------------
Four choices for wiring diagrams:
* `ExecutionLink` -- function-call-like
* `RuleLink` -- inference-like
* `FilterLink` -- electronics-circuit-like
* `Section` -- grammatical/sheaf-like

Pros and cons:

### EvaluationLink
Olde-school. Uses  `GroundedSchemaNode` for the function name, and a
list of inputs, but no defined outputs. Can be given a TV to indicate
probability, but no clear-cut interpretation of the function arguments.
Replaced by `EdgeLink` for performance/size.

### RuleLink
Originally intended to be a base class for rewriting (for PLN).

* Fairly compact.
* Distinct "input" and "output" areas.
* Rules are nameless (anonymous) with no built-in naming mechanism.
* Rules are explicitly heterosexual (everything is either an input, or
  an output, and must be one or the other.) This is problematic if
  inputs or outputs are to be multiplexed, or given non-directional
  (monosexual) conntions.
* No explicit connection/glueing semantics.

`RuleLink`s specify term rewrites. Ideal for forward-chained rewriting.
With great difficulty, can be backwards-chained. That this was a difficulty
was exposed by PLN development.

### FilterLink
The `FilterLink` is used to specify a filter that can be applied to a
stream.  The stream can be Atoms or Values or both. (This distinguishes
it from the `QueryLink`, for which the source and target must lay within
the AtomSpace itself.)

Wrapping a `RuleLink` with a `FilterLink` is an effective for specifying
rewrite rules to be applied to the stream that the filter is filtering.

Problem is that this notion is very fine-grained. The filter can accept
one or more streams as input, apply a rewrite, and generate one or more
outputs. The inputs and the outputs are streams, but to specify the
"wiring diagram", in Atomese, those streams are necessarily anchored
at well-known locations (typically, under some key on some
`AnchorNode`.)

This makes processing with FilterNodes resemble an electronics circuit
diagram. Each "dot" in the circuit is some Atom-with-Key location, that
must be explicitly specified. Each circuit element, spanning two or more
dots, is a `FilterLink` + `RuleLink` combo. The connection to a "dot"
is done with a `ValueOfLink` that explicitly names the dot.

Async elements, such as `QueueValue`s exist and work. This allows
readers to stall and wait for input. mux and demux are not an issue:
`QueueValue`s automatically demux, and a mux can be created with a
RuleLink that simply copies one input to two (or more) places.

Arbitrary processing DAG's can be built in this way. Running on the host
CPU. The proximal question is: can this be converted into a system for
wiring up flows on GPU's?

### Sections
Sections and connectors were originally invented to generalize the
concept of tensors, by allowing the specification of arbitrary
star-shaped patterns aka semi-assembled jigsaws. The types of the
connectors are loosened: instead of having just two types, "input"
or "output", there can be any number of types. The connection rules
are loosened as well: instead of assuming "heterosexual" rules, where
outputs must connect to inputs (only), one can have arbitrary mating
rules. Thus, a connection is allowed, if the mating rules allow it:
typically, the types must match, and the "sex" of the connectors must
be compatible.

As abstract mathermatics, this is a very powerful abstraction. As a
programming API, it is verbose and hard to work with. In particular,
a wiring engine is needed. Such a wiring engine has not yet been
created.

It is not particularly hard to specify a GPU kernel as a sheaf section,
but perhaps verbose. But that just specifies (describes) it. It also
needs to be wired in place. How?

Design Requirements
-------------------
For a prototype proof-of-concept demo, what's actually needed?

* **Basic vector I/O to GPU's.**
  Minimal function is to move data to GPU's, call kernel, get data back.

* **Multi-step dataflow wiring.**
  Define sophisticated data flow wiring. Similar to TensorFlow (???)

* **AtomSpace on GPU.**
  Run AtomSpace on GPU. Use `StorageNode` API to talk to it.

For bootstrapping, lets stick to the basics.

### Basic vector I/O
Ability to stream data data to GPU. One-shot is a special case.
Pseudocode:
```
   ; Location of kernel code as path in local filesystem.
   ; OpenCLNode isA SensoryNode
   (SensoryNode "opencl://platform:device/file/path/kernel.cl")
   (SensoryNode "opencl://platform:device/file/path/kernel.clcpp")
   (SensoryNode "opencl://platform:device/file/path/kernel.spv")

   ; Apply standard Open to SensoryNode
   ; Returns stream handle, which must be stored at some anchor.
   (Open
      (Type 'OpenclStream)
      ((Sensory "opencl://Clover:AMD/tmp/vec-mult.cl"))

   ; Write two vectors to GPU. Apply 'vect_mult' kernel function to
   ; the vectors. The 'List' will typically be a 'ListValue'.
   (WriteLink
      (ValueOf anchor)
      (List
         (Predicate "vect_mult")
         (Number 1 2 3 4)
         (Number 2 2 2 2)))

   ; Place output stream at well-known location
   (cog-set-value!
      (Anchor "some anchor point")
      (Predicate "some key")
      (StreamValue "0 0 0"))

   ; Copy results from GPU to AtomSpace.
   (ReadLink
      (ValueOf anchor)
      (ValueOf (Anchor "some anchor point") (Predicate "some key")))

   ; TODO RUleLink + FilterLink that provides an all-in-one wrapper for
   ; above. See sensory examples for example.
```
