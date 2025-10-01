Design Notes G
==============
September 2025

Notes about the mechanics of running AST's on GPU's.

The base implementation of the GPU interfaces is now sufficiently
advanced that the next steps open up: the ability to take a given
Atomese AST and run it on a GPU. That is, to create a GPU-cenetric
library of arithmetic functions, and then take a specific abstract
syntax tree (AST) written as e.g. `(AccumLink (TimesLink A B))` and
run it on the GPU. But what is the API to this, and what is the
mechanics of doing this?

One possibility is to have PlusGPULink and TimesGPULink, and then have
RuleLinks to transpose the AST from its "abstract" form to these
concrete Atoms. This is problematic because RuleLinks remain difficult
to work with and apply. They seem like a great idea but are tough.
There are also little details that threaten coherency: stuff that might
need to run locally before getting pushed off to the GPU, and the rules
for these could be very arcane and detailed and prone to bugs.

Another possibility is to have an `(ExecOnGPULink AST)`, similar to
`(PureExec AST)`. In fact, it could be PureExec, given extensions.
Lets review.

`PureExec` was originally invented to allow "pure execution" i.e.
side-effect free execution of some AST by grabbing a temp AtomSpace,
child of the current AtomSpace, and running the AST there. Any changes
to the AtomSpace are thus made only to this temp space, and are thus
discarded when execution completes. Thus, the execution is "pure".

Next `PureExec` got used to define tail-recursive loops. The structure
here is
```
   (DefineLink (DefinedProcedure "foo")
      (PureExec AST (DefinedProcedutre "foo")))
```
so the AST performs a single-step of the loop, and the
`DefinedProcedure` recurses. The `CondLink` can be used to limit
recursion.  This is more or less just like tail recursion on scheme.

Note that an earlier realization of this idea was with the behavior
trees infrastructure, with `SequentialAndLink`. This used evaluation,
not execution, and the `SequentialAnd` had a built-in "keep going or
stop" character in it, that avoided the need for an explicit `CondLink`.
It's inappropriate for execution, because the returned Value is
generally not interpretable as a bool flag "keep going".

Until `PureExec` was introduced, there wasn't anything analogous for
execution. The `ParallelLink` and cousins run threads, but there was no
`SerialLink` mostly because step-wise execution was not needed.

PureExec needs to be redesigned as described below.

Since the initial idea of `PureExec` was purity, it could not be used
for tail-recursive, serialized execution in the current AtomSpace: so an
argument is added: which AtomSpace the execution.  The proposal here is
to generalize this to allow a GPU to be specified, or more generally an
external object.

But first, a distraction:
The `ExecuteThreadedLink` gathers results into a QueueValue.  The
`ParallelLink` fires and forgets: it runs threads, dettaches, does
not wait for completion; does not collect results. Also does not specify
max-threads to use (because doing so des not seem to make sense, if
one is not going to wait?)

TODO List
---------
Writiing the above, the TODO list expands:
* The ThreadJoinLink should be removed from Atomese. DONE.
