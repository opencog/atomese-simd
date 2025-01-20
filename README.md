
Atomse SIMD (OpenCL/CUDA) Interfaces
====================================
Experimental effort to enable I/O between Atomese and SIMD compute
resouces. The primary target is commodity GPUs, running either OpenCL
or CUDA. The current prototype accesses the hardware at a low level,
working with individual vectors. High-level API's, such as commonly
used in deep learning, are not supported in this interface layer.

[Atomese](https://wiki.opencog.org/w/Atomese), the interface language for
the OpenCog [AtomSpace](https://github.com/opencog/atomspace) hypergraph
database, has a variety of different ways of talking to external
subsystems. These include:

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
* Obsolete gateways to ROS, the Robot Operating System, to Minecraft
  (via MineRL & Malmo), to Unity, the game engine, and many more.

Interfacing to GPU subsystems, such as CUDA or OpenCL, or any of a large
variety of systems built on these, such as TensorFlow, offer a
non-trivial exercise for testing and guiding the Atomese sensori-motor
interfaces.

The interfaces here simply move vectors to and from GPUs and invoke GPU
kernels to perform processing on them. Perhaps high-level interfaces
will be added in the future; they're not here in this prototype.

The main effort is to find a balance between abstract mathematical
theory and a practical, usable interface. The generic Atomese
sensorimotor research is being done in the hopes of discovering
generic abstractions that symbolic (and neuro-symbolic) AI subsystems
can use. If you are a human programmer, you have a zilllion-and-one
choices for programming languages and APIs that allow you to hack up
almost anything. If you are a symbolic AI agent, or a DL/NN transformer,
you do not have this richness of tools. You mostly don't even have much
of a clue of what "reality" is. In part, that's because you don't have
a sensori-motor system, beyond some hacked-up robots and in-game avatars
created for you by *human* engineers. Perhaps with a proper theory, some
of the hackiness can be refined.

The need for a general sensori-motor theory is underscored by the
failure of earlier OpenCog projects that attempted create "embodied
OpenCog agents". These are the systems listed above: interfaces into
ROS, Unity and Minecraft, to name some of the more advanced efforts.
These were all sensori-motor "hack jobs", built without giving any
thought of what it means "to perceive and move".

Status
-----
***Version 0.0.6.*** --
Basic proof-of-concept, showing how to use Atomese to open a connection
to an OpenCL compute device (i.e. a GPU), load and invoke compute
kernels, and have those kernels work with floating-point vector data
residing in the AtomSpace.

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
  Maybe more; depends on your distro and hardware.
  (Maybe also: `ocl-icd-opencl-dev` and `opencl-clhpp-headers`?)
* For Nvidia devices, install `cuda-opencl-dev`
  Maybe more; depends on your distro and hardware.
* Optional: Install `clang-14` and `llvm-spirv-14`
  Demos can use "offline-compiled" (pre-built) kernels.
* `sudo usermod -a -G video <user_id>`
* Build and install cogutils, the AtomSpace, sensory and then the code
  here.  This uses the same build style as all other OpenCog projects:
  `mkdir build; cd build; cmake ..; make; sudo make install`
* Run unit tests: `make check`.  If the unit tests fail, that's proably
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
