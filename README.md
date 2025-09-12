
Atomese SIMD (OpenCL/CUDA) Interfaces
====================================
Experimental effort to enable I/O between Atomese and SIMD compute
resources. The primary target is commodity GPUs, running either OpenCL
or CUDA. The current prototype accesses the hardware at a low level,
working with individual vectors. High-level API's, such as commonly
used in deep learning, are not supported in this interface layer.

The Experiment
--------------
The experiment, as it is currently evolving, is to understand graph
rewriting between "natural" descriptions of compute systems. For
example, the "natural" description of a deep learning transformer is
a kind of wiring diagram, indicating where data flows. By contrast, the
"natural" description of GPU compute kernels are C/C++ function
prototypes. How can one perform the rewrite from the high-level
description to the low-level description? Normally, this is done "brute
force": a human programmer sits down and writes a large pile of GPU
code, creating systems like PyTorch or TensorFlow. The experiment here
is to automate the glueing together from high-level to low-level
descriptions.

Such automation resembles what compilers do. For example, a C++ compiler
accepts C++ code and issues machine assembly code. A Java compiler
accepts Java and issues Java bytecode. The goal here is to accept
high-level functional descriptions, such as that of a transformer, and
to generate low-level invocations of GPU kernels. However, the project
here is not meant to be a "compiler" per se; if that is what one really
wanted, one could just brute-force write one (or use PyTorch,
TensorFlow, ...) Instead, this is meant to be an exploration of graph
rewriting in general, and the GPU target just happens to be a realistic
and suitably complicated example target.

A different way of thinking of this experiment is as an exploration of
the agency of sensori-motor systems. The GPU is a "thing out there" that
can be manipulated by an agent to "do things". How does one perceive the
"thing out there", and describe it's properties? How can one control it
and perform actions on it? To extert motor control on that "thing out
there"? How does one perceive the results of those motor actions?

In the present experiment, the "agent" is is a collection of graphs
(properly, hypergraphs), represented using
[Atomese](https://wiki.opencog.org/w/Atomese),
and "living" in the OpenCog
[AtomSpace](https://github.com/opencog/atomspace).
That is, the subjective inner form of the agent is a world-model,
consisting of abstractions derived from sensory perceptions of the
external world, and a set of motor actions that can be performed to
alter the state of the external world.  The agent itself is constructed
from Atomese; the GPU is a part of the external world, to be manipulated.

The above description might feel like excessive anthropomorphising of
a mechanical system. But that's kind of the point: to force the issue,
and explore what happens, when an agentic viewpoint is explicitly
forced.

External Subsystems
-------------------
[Atomese](https://wiki.opencog.org/w/Atomese) is the interface language
for the OpenCog [AtomSpace](https://github.com/opencog/atomspace)
hypergraph database. It has a variety of different ways of talking to
external subsystems. These were developed to solve various practical
application issues that arose in the use of Atomese. These subsystem
interfaces are "well designed" in that they solve the particular issue
that arose. They are not, however, explictly agentic or sensori-motoin
their design. Some shadows and hints of this can be seen, however.

These external subsystem interfaces include:

* The [GroundedSchemaNode](https://wiki.opencog.org/w/GroundedSchemaNode)
  allows external python, scheme and shared-library functions to be
  called, passing arguments encoded as Atoms.
* The [StorageNode](https://wiki.opencog.org/w/StorageNode) allows Atoms
  to be sent to and received from various locations, including Internet
  hosts (using
  [CogStorageNode](https://wiki.opencog.org/w/CogStorageNode)), disk
  drives (using
  [RocksStorageNode](https://wiki.opencog.org/w/RocksStorageNode)),
  databases (using
  [PostgresStorageNode](https://wiki.opencog.org/w/PostgresStorageNode)),
  files (using
  [FileStorageNode](https://wiki.opencog.org/w/FileStorageNode))
  and more.
* The [SQL Bridge](https://github.com/opencog/sql-bridge) allows
  SQL Tables to be mapped into AtomSpace structures, so that updates
  to one are reflected as updates to the other.
* Obsolete gateways to ROS, the Robot Operating System, to Minecraft
  (via MineRL & Malmo), to Unity, the game engine, and many more.
  These can be found in old, archived
  [github repos](https://github.com/opencog/), scrolling down to
  the oldest repos having no activity.

The above "work", but lack an agentic sensori-motor design interface.
To use these interfaces

* The [Sensory](https://github.com/opencog/sensory) system, which is
  an experimental effort to understand the generic mathematical theory
  of interfacing Atomese to arbitrary unknown sensory devices, and to
  able to use them to obtain streams of data. This includes the
  abstract definition of a "motor", which is a device that can cause
  changes to the external world (such as movement or manipulation of
  objects).
* The [Motor](https://github.com/opencog/motor) system, which attempts
  to define a simpler, more practical and mundane way of using Atomese
  to work with external devices.

Status
------
***Version 0.1.0*** --
Basic proof-of-concept, showing how to use Atomese to open a connection
to an OpenCL compute device (i.e. a GPU), load and invoke GPU compute
kernels, and have those kernels work with floating-point vector data
that is moved between the AtomSpace and the GPU.

New in this version: Atomese interface descriptions are now generated
for OpenCL kernel interfaces. This should allow introspection of the
interfaces, and their manipulation in Atomese. We'll see how that goes.
Some problems are already visible. But some groundwork is there.

The demo is minimal, but it works.  Tested on both AMD Radeon R9 and
Nvidia RTX cards.

Overview
--------
The directory layout follows the conventional AtomSpace standards.

Notable content:

* The [examples](examples) directory contains a working example
  of Atomese interacting with a GPU.
* The [scaffolding](opencog/opencl/scaffolding) directory
  contains some bring-up code and several hello-world examples.
* [Design Notes](Design.md) contains some
  raw ideas on how the system should be (and was) designed.
* The [types](opencog/opencl/types) directory contains
  definitions for some OpenCL Atom types.
* The [atoms](opencog/atoms/opencl) directory contains
  implementations for those Atom types.


HOWTO
-----
Steps:
* Get some OpenCL GPU hardware. The demo should work on anything and has
  been tested on a Radeon graphics card and an Nvidia card.
* Install `clinfo`.
* For AMD devices, install `mesa-opencl-icd` and `opencl-headers`
  and `opencl-clhpp-headers` and `ocl-icd-opencl-dev`
  Maybe more; depends on your distro and hardware.
* For Nvidia devices, install `cuda-opencl-dev`
  Maybe more; depends on your distro and hardware.
* Optional: Install `clang-14` and `llvm-spirv-14`
  Demos can use "offline-compiled" (pre-built) kernels.
* User *must* have permission to write the device(s) located in
  `/dev/dri`; typically `/dev/dri/renderD128` or similar.
```
    sudo usermod -a -G video <user_id>
    sudo usermod -a -G render <user_id>
```
* Build and install cogutils, the AtomSpace, sensory and then the code
  here.  This uses the same build style as all other OpenCog projects:
  `mkdir build; cd build; cmake ..; make; sudo make install`
* Run unit tests: `make check`.  If the unit tests fail, that's probably
  because they could not find and GPU hardware. See below.
* Look over the examples. Run them
  `cd examples; guile -s atomese-kernel.scm`

### If unit tests fail
The `scaffolding` directory contains code that is "pure" OpenCL,
and does not have any Atomese in it. It just provides some basic
OpenCL examples. Runs on both AMD and Nvidia hardware.

Make sure the software isn't insane, by running the
`opencog/opencl/scaffolding/show-ocl-hw` executable from the
`build` directory. It will print a short hardware listing that
is a subset of what the `clinfo` command lists. If it doesn't
work, that is probably because it's too stupid to find your hardware.
Read the source, Luke.

Make sure you can talk to the hardware, by running the
`opencog/opencl/scaffolding/run-hello-world` executable from
the `build` directory. It should print `>>This is only a test<<` if
the code ran on the GPUs.  It will work only if there is a copy of
`hello.cl` in whatever directory that you are running `run-hello-world`
from.

The `opencog/opencl/scaffolding/run-vec-mult` executable is
similar to above; it performs a simple vector multiply.

The `run-flow-vec` executable is a rework of above, to more clearly
define and prototype the distinct steps needed to flow data.
