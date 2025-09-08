Design Notes B
==============
August 2025

Notes about OpenCL interfaces and how they impact Atomese design.

`OpenclNode` currently holds:
```
   cl::Platform _platform;
   cl::Device _device;
   cl::Context _context;
   cl::Program _program;
   cl::CommandQueue _queue;
```
Jobs are queued to it using `job_t` which holds:
```
   cl::Kernel _kernel;
   std::vector<cl::Buffer> _invec;
   cl::Buffer _outvec
```

I want to wrap `cl::Buffer` with either a `Value` or a `Node`. Probably
`Value` is best; it sticks to the original vision for mutable Values.
But an `ObjectNode` (or `SensoryNode`) interface is not outrageous.
Lets review,


### OpenclFloatVecNode
Derived from `SensoryNode`. Thus has the mndatory set of
open/close/read/write methods.  Allows explicit control.

For example: upload vector data to GPU:
```
(SetValue
	(OpenclFloatVecNode "some name")
	(Predicate "*-write-*")
	(Number 1 2.71828 3.14159 42))
```
Except we might not want to upload just then but simply bind the
float data to the object. Hmmm.

Download vector data from GPU:
```
(GetValue
	(OpenclFloatVecNode "some name")
	(Predicate "+-read-*"))
```
Optional: specify vectors to be `FloatValue`s or `NumberNode`s.
```
(SetValue
	(OpenclFloatVecNode "some name")
	(Predicate "*-open-*")
	(Type 'FloatValue)
```
If not specified, defaults to `FloatValue`

Issues:
* The `cl::Buffer()` ctor needs `cl::Context` as an argument. The
  `cl::Context` only becomes available after the platform and device
   are found. This happens in `OpenclNode::open()`.

Solution: leave unbound until an explicit `*-bind-*` message, or until
an implicit bind because it gets used.

### OpenclFloatValue
Traditional Value design. Referencing it calls `update()` which
downloads from the GPU, if that has not already been done. Carries
additional protected methods that allows OpenclNode to work with
buffers and bind them as needed.

Usage: the existing demo `atomese-kernel.scm` is effectively unaltered.
Lets try this. I think it will work cleanly.

### Jobs
The natural GPU job seems to tie together a `cl::Kernel`	with the
vectors it takes as input and generates as output. Current prototype
writes:
```
      (Section
         (Predicate "vec_mult") ; Must be name of kernel
         (ConnectorSeq
            (Type 'Number)
            (Number 1 2 3 4 5)
            (Number 2 2 2 2 2 2 3 42 999)))))
```
but it makes to replace `Section` by an `OpenclJobValue` (which would
be of type `SectionValue`, I guess.) and then have `OpenclJobValue`
manage the kernel.  This architecture then starts to strongly resemble
the `ExecutionOutputLink` architecture, with assorted differences:
the traditional `ListLink` is replaced by `ConnectorSeq` so that
both inputs and outputs can be encoded in the same flat format.
Of course, ExOutLink could be modified to do the same. And, of course,
```
(cog-execute! (ExecutionOutputLink ...))
```
is replaced by
```
(cog-execute! (SetValue (OpenclNode ...) *-write-* (Section ...)))
```
i.e. any kind of method.

So this makes the `Section` not directly ececutable, but requiring
a hand-off to a system that knows how to execute it. We could redo
the ExOutLink to follow this style, and write
```
(cog-execute! (SetValue (PythonNode ...) *-write-* (Section ...)))
```
Its not vlear that this has any actual avantages to the current
ExOutLink for FFI's like python.

-------
