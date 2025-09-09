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
Let's recap the grand structuralist idea. There are a set of jigsaw
pieces; each resembling a Link Grammar disjunct. The connector rules
define how these can be connected. Ultimately, they are meant to be
self-assembling; all prototypes to date have not reached this stage
(except for the very old language demo).

The result of connecting jigsaws is a wiring diagram. This is much
like electronics: a transistor has three attachment points, and a
circuit is valid only if all three are connected. A different example
arises in GCC's gimple: each insn has a set of connectors, defining
input, output, state, widths, side effects, and the goal of the
compiler is to generate a valid net.

In both examples, the resulting network can process flows; for
eletronics, the flows consist of electrons; for programs, it is bytes
passing between registers, CPU and memory.

The current sensory demos use a `FilterLink`/`RuleLink` combo to
represent a processing element. The `DescribeLink` aka `*-describe-*`
message was meant to provide a description of the connectors; this
has not been implemented yet.


----
