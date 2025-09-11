Design Notes D
==============
September 2025

Implementation notes. The late-August 2025 version of the code had
assorted conceptual design issues. These are documented below, with
a discussion of design alternatives. All of the suggested solutions
have been implemented as of 11 September 2025. ... More or less
as described, with some variation.

Kernel Execution
----------------
What happens when
```
    (cog-execute! (OpenclNode ...) (Predicate "*-write-*")
        ( ... data ...))
```
is sent? Well, the `*-write-*` method has to validate that the data
is actually of the format defined in the `Section` declaration. If not,
it throws an error. (We throw, instead of silently failing, to ease
debugability.)

Here's the logic for the (implicit) type conversions in the data stream:

 * If an input vector is a `NumberNode` or a `FloatValue`, a
   corresponding `OpenclFloatValue` is created, copying the vector
   out of the `NumberNode` or `FloatValue`, and marking it as
   `CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR`.

 * If an input vector is already an `OpenclFloatValue`, nothing needs
   to be done, except maybe to or-in `CL_MEM_COPY_HOST_PTR` ??

 * If an output vector is already an `OpenclFloatValue`, nothing needs
   to be done, except maybe to or-in `CL_MEM_READ_WRITE`.

The only problem I see here is that these `CL_MEM_*` flags are set up
by the `cl::Buffer()` ctor; can they be changed later? Hmm. Well,
certainly `enqueueReadBuffer` can specify the `void*` data pointer at a
later time. Also `enqueueWriteBuffer` allows the data location to be
deferred, and so `CL_MEM_COPY_HOST_PTR` does not need to be specified at
ctor time.

A `cl::Buffer` created read-only cannot be converted to read-write.

The current demo allows the output vector to be specified as a
`(TypeNode 'FloatValue)`, after which it implicitly creates the
required `OpenclFloatValue` and sets it to the correct size, and then
returns it with the `*-read-*` method. This is OK, I guess, unless there
are two outputs, in which case `*-read-*` needs to return a `LinkValue`
holding both.

Which raises an old issue: due to the async nature, the `*-read-*`
should return not just the outputs, but the entire wrapped kernel.
A later read can retrieve individual outputs, as required.

OK, I think that's a plan. Lets do it.

Bug: Which kernel?
------------------
The actual `cl::Kernel` depends on the `cl::Program`, which is currently
managed by the `OpenclNode`. If there were two different `OpenclNode`s
with two different programs, but having some of the same names for the
kernels, then there's a clash, because `OpenclKernelNode` would get
bound to the first one.

One solution is to encode the full URL into the kernel name. Thus:
```
    (OpenclKernel "opencl://Clover:AMD Radeon/some/where/some-prog.cl?kern_name")
```
but this is a very long string that forces URL processing out of scope.
It works, it's unambiguous, its annoyingly verbose and possibly
error-prone. It also encourages storing the `OpenclNode` inside the
`OpenclKernelNode` which turns it into a defacto link type thing. Yuck.

The only clean way of saying "do this in this-and-such a context" in
Atomese is to use a `Link`. This suggests the following:
```
    (OpenclKernelLink
        (Opencl "opencl://Clover:AMD Radeon/some/where/some-prog.cl")
        (Predicate "kernel_name"))
```
The super-long URL is now split in two, and the context is clearly
established.  The only drawback here is that the write message becomes
dorky:
```
    (cog-set-value!
        (Opencl "opencl://foo.cl")
        (Predicate "*-write-")
        (Section
            (OpenclKernelLink
                (Opencl "opencl://foo.cl")
                (Predicate "kernel_name"))
            ...))
```
The `OpenClNode` gets specified twice. The second feels superfluous,
because it's indirectly knowable from the target of the write. But
still...

A different issue is that the the `OpenclNode` could be sent the
`*-close-*` message. This would require it to do a `getIncomingByType`
and notify each of the `OpenclKernelLink` in it's incoming set that they
should close as well. Kind of icky, but it works.

What should `OpenclKernelLink` inherit from? It could inherit from
`Connector` (gasp!) but that seems to muddle what a `Connector` is
suppose to be. Perhaps `TagLink` ...

I don't see any other ways that are reasonable and consistent.

(Update: Above was a bad idea and it was removed. The kernel needs to
be bound up with the arguments that it takes. A new instance of
`cl::Kernel` needs to be created for each compute job, as otherwise
we risk clobbering the `cl::Kernel::setArg()` as we move along. See
proposed solution immediately below.)

Bug/Issue: Actions/Operations
-----------------------------
The above design has another flaw: it binds a single instance of
`cl::Kernel` to `OpenclKernelLink`. That's wrong; it should be a single
instance of `cl::Kernel` per job, with each job having different
inputs and outputs. What's needed is something like `OpenclSectionValue`
to bind together one instance of `cl::Kernel` with all the calls to
`cl::Kernel::setArg()` to tie thhem all together. Once all bound up,
then it can be launched.

Several sub-issues: (1) who creates it? (2) How can user get a handle to
it, in case they want to run it repeatedly? (3) is there a better name
than `OpenclSectionValue`? Something like "run job" or "do operation"
or "perform action", so maybe `OpenclActionValue` ...

* Call it `OpenclJobValue`.
* Use `*-write-*` `Section` to digest the `Section` and convert it to an
  equivalent `OpenclJobValue`.
* Run it. The first `*-read-*` returns the `OpenclJobValue`.
* Sending `*-write-*` `OpenclJobValue` runs it. No return value.
* Drop support for `SectionValue`.
* Git rid of `OpenclKernelLink`.

Update: Above is implemented. Writes always return the `OpenclJobValue`
as a status indicator. This seems unavoidable, if we want to block and
wait for a job to be done.

----
