Design Notes B
==============
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
open/close/read/write methods.  Alllows explicit control.

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
addtional protected methods that allows OpenclNode to work with
buffers and bind them as needed.

Usage: the existing demo `atomese-kernel.scm` is effectively unaltered.
Lets try this. I think it will work cleanly.


-------
