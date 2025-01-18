
Atomse OpenCL Interfaces
========================
Experimental effort to enable I/O between Atomese and OpenCL devices
(and, in particular, GPUs).

[Atomese](https://wiki.opencog.org/w/Atomese), the interface language for
the OpenCog [AtomSpace](https://github.com/opencog/atomspace) hypergraph
database, has a variety of different ways of talking to external
subsystems. These include:

* [GroundedProcedureNode](https://wiki.opencog.org/w/GroundedProcedureNode)
  allows external python, scheme and shared-library functions to be
  called, passing arguments encoded as Atoms.
* [StorageNode](https://wiki.opencog.org/w/StorageNode) allows Atoms to
  be sent to and received from various locations, including internet
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
  abstract defintion of a "motor", which is a device that can cause
  changes to the external world (such as movement or manipulation of
  objects).
* The [Motor](https://github.com/opencog/motor) system, which attempts
  to define a simpler, more practical and mundane way of using Atomese
  to work with external devices.
* An obsolete gateway to ROS, the Robot Operating System.

Interfacing to GPU subsystems, such as CUDA or OpenCL, or any of a large
variety of systems built on these, such as TensorFlow, offer a
non-trivial exercise for testing and guiding the Atomese sensori-motor
interfaces.

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


Status
-----
***Version 0.0.2.***
Some prototyping, some demos, some unfinished ideas for design &
implementation. Nothing connects with Atomese, yet.

Overview
--------
Directory layout:

* The [scaffolding](scaffolding) directory contains some bring-up code
  and hello-world examples.
* The [opencl-types](opencl-types) directory contains definitions for
  some OpenCL Atom types.
* The [stream](stream) directory contains implementations for those
  Atom types.


HOWTO
-----
Steps:
* Get some OpenCL GPU hardware, such as a Radeon graphics card.
* Install `clinfo` and `mesa-opencl-icd` and `opencl-headers`
  Maybe more, for your hardware.
* Maybe also: `ocl-icd-opencl-dev` and `opencl-clhpp-headers` ?
* `sudo usermod -a -G video <user_id>`

Make sure the software isn't insane, by running
`opencog/opencl/scaffolding/show-ocl-hw` executable from the `build`
directory. It will print a short-form hardware listing that should
match what the `clinfo` command lists. If it doesn't, something is
wrong with the code here.

Make sure you can talk to the hardware, by running the
`opencog/opencl/scaffolding/run-hello-world` executable from the `build`
directory. It should print `>>This is only a test<<` if the code ran
on the GPUs.  It will work only if there is a copy of `hello.cl` in
whatever directory that you are running `run-hello-world` from.

The `opencog/opencl/scaffolding/run-vec-mult` executable is similar
to above; it performs a simple vector multiply.
