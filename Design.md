Design Notes
============
The design of the system is being developed in a sequence of texts,
articulating issues and possible solutions. The overall task is to
explore the structuralist approach to knowledge representation. A recap
of what this is is given further below.

This is a design diary, and earlier notes may not reflect the current
actual design.

* [Design-A](./Design-A.md) provides notes about OpenCL interfaces and
  how they impact Atomese design.
* [Design-B](./Design-B.md) examines the Sensory Version Half port.
* [Design-C](./Design-C.md) reviews wiring, composability and flow.

Structuralism Overview
----------------------
The Grand Structuralist Idea is that *everything* can be represented in
terms of (hyper-)graphs, and that there are informational flows on the
edges of these graphs.

The graphs are to be (self-)assembled from jigsaw pieces. The connector
rules define how these can be connected. Ultimately, they are meant to
be self-assembling, and thus should operate at a phase transition
critical point. None of the prototypes have reached this stage, except
for the old language-learning project that inspired this effort.

In that project (the language-learning project), the jigsaws were
Link Grammar disjuncts. These were learned by observing text and
counting. The LG parser provide the assembly mechanism; they were "self
assembling" in that given a dictionary of disjuncts, grammatically valid
sentences come out.

In the general case, the result of connecting jigsaws is a wiring diagram.
This is much like electronics: a transistor has three attachment points,
and a circuit is valid only if all three are connected. A different example
arises in GCC's gimple: each insn has a set of connectors, defining input,
output, state, widths, side effects, and the goal of the compiler is to
generate a valid net.

In both examples, the resulting network can process flows; for eletronics,
the flows consist of electrons; for programs, it is bytes passing between
registers, CPU and memory.

The present project, `atomese-simd`, explores how to create jigsaws that
can be used to describe GPU software kernels, and how to interact with
the GPU to perform computation. One of the goals is to allow DL/NN
networks, such as transformers, to be created from Atomese descriptions.
That is, the wiring diagrams are meant to represent typical DL/NN type
processing networks.

This project is challenging, because it requires the ability to
represent external state (the state of the GPU) and to be able to
manipulate it (upload/download vectors; add and multiply vectors,etc.)
It is sensori-motor in nature, in that the external state is sensed
(by downloading it) and acted on by 'motors' (the GPU kernels). The
kernels are assemblies of jigsaws: of primitive processing elements,
such as addition, multiplication or sigmoids.

Of course, human programmers, as well as Claude, can already do this,
using conventional systems like PyTorch or TensorFlow or whatever. The
goal here is to have a representation that is dynamic (can be created on
the fly) and accessible to direct manipulation. At this time, Claude is
not yet capable of managing the architectural complexity to do this, and
the conventional software stack lacks the correct properties. So
low-level Atomese design and research seems to be the only way forward.

----
