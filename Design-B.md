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

I want to wrap `cl::Buffer` with either a `Value` or a `Node`. It seems
that an `ObjectNode` (or `SensoryNode`) interface is best. For example:

Upload vector data to GPU:
```
(SetValue
	(OpenclFloatVecNode "some name")
	(Predicate "*-write-*")
	(Number 1 2.71828 3.14159 42))
```

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


-------
