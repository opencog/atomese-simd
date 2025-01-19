
Atomse OpenCL Interfaces
========================
Experimental effort to enable I/O between Atomese and OpenCL devices
(narrowly, to GPUs; broadly, to DL/NN architectures.)

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

At the narrowest level, the interfaces here simply move vectors to and
from GPUs and invoke GPU kernels to perform processing on them. More
broadly, the interfaces here explore some of the more complex APIs
commonly used by major systems to communicate data across channels and
process it on remote servers (such as GPUs).

The effort here attempts to find a balance between abstract
mathematical theory and a practical, usable interface. The generic
Atomese sensorimotor research is being done in the hopes of discovering
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
***Version 0.0.2.*** --
Some prototyping, some demos, some unfinished ideas for design &
implementation. Nothing connects with Atomese, yet.

Overview
--------
The directory layout follows the conventional AtomSpace standards.
Everything exciting is in the
[opencog/atoms/opencl](opencog/atoms/opencl) directory.

Notable content:

* The [scaffolding](opencog/opencl/scaffolding) directory
  contains some bring-up code and several hello-world examples.
* [Design Notes](Design.md) contains some
  raw ideas on how the system should be designed.
* The [types](opencog/opencl/types) directory contains
  definitions for some OpenCL Atom types.
* The [stream](opencog/atoms/opencl/stream) directory contains
  implementations for those Atom types.


HOWTO
-----
Steps:
* Get some OpenCL GPU hardware, such as a Radeon graphics card.
* Install `clinfo` and `mesa-opencl-icd` and `opencl-headers`
  Maybe more; depends on your distro and hardware.
* Maybe also: `ocl-icd-opencl-dev` and `opencl-clhpp-headers` ?
* Optional: Install `clang-14` and `llvm-spirv-14`
  Demos can use "offline-compiled" (pre-built) kernels.
* `sudo usermod -a -G video <user_id>`
* Build and install cogutils, the AtomSpace, and the code here.
  This uses the same build style as all other OpenCog projects.

Make sure the software isn't insane, by running
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
