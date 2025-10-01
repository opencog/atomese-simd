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

### Execute ... where?
What should the argument to `PureExec` be? Currently, it allows an
AtomSpace to be specified. If none specfied, an anonymoous temporary
child AtomSpace is used. Otherwise, it is implicitly assumed that
the specified AtomSpace inherits from the current AtomSpace, as
otherwise, bad things are likely to happen. Untested: what does it
even mean to execute "somewhere else"?

* The Atoms appearing directly in the PureExec would need to be
  copied to the remote locaton. (This is not needed when the "remote"
  location is a child of the current AtomSpace)

* There is explicit mechanism for copying Atoms around, other than
  what is provided by `StorageNode`

* Should remote AtomSpaces be a kind-of StorageNode? Conversely,
  perhaps StorageNode should be a kind-of AtomSpace?

* Currently, remote execution is provided by `fetch-query` in the
  StorageNode; presumably this can/should be extended to be any
  kind of execution!?

The above questions envision an extension/mutation of the current
StorageNode concepts to be AtomSpacee-like. Which seems like an awesome
idea, as I write this.

### Frames and Proxies
The layering of AtomSpaces was inspired by `git` and was envisioned as
a directed acyclic graph, with each AtomSpace above inheriting from the
AtomSpaces below. The implicit, unspoken assumption was that all of
these were local, on the same system/RAM, and thus directly accessible.
Because everything is local, another implicit assumption is that both
reads and writes are described by this graph: an AtomSpace accessible
for reading is also accessible for writing (unless it is marked
read-only.)

The current implementation of layered AtomSpaces has some strong
negative performance reprecussions. This is because the layers are
envisioned as being OverlayFS-like: Atoms in lower layers can be masked
or hidden in higher layers, so that it looks like they are absent.
Atoms in higher layers can have different key-value store contents
than lower layers.  The lookup has fairly strongly noticable performance
impacts, even though this is not benchmarked at this time.

This directed graph is represented with a Frame, which is an Atom that
is both like a Node and like a Link, having both a (singular) name and
an outgoing set.

A completely different layering mechanism is that of the `ProxyNode`,
which is a kind-of `StorageNode`. It replaces the idea of a DAG with
a more flexible system:

* Reads are distinct from writes; one can be handled, and the other not.
* Dropping writes makes the proxy effectively read-only.
* Read priority and caching are possible.
* Write mirroring and write sharding are possible.

Some aspects are missing: ProxyNodes can only supplement AtomSpace
contents; they do not provide any ability to hide existing Atoms,
or to present an alternate view of an Atom, in the way that the layers
do. The layering system provides a "view" of the AtomSpace. It has
many of the same problematic issues that OverlayFS does, namely,
complexity and performance.

This suggests that Frames should be replaced by ProxyNodes. Which again
elevates a StorageNode (this time, in the form of a Proxy) to be at the
same conceptual level as the AtomSpace.  So we now have three distinct
"remote" operations: read, write and execute.

### Reinventing Cmputing?
The `rwx` certainly make it feel like we're re-inventing computing.
Lets take a closer look and make sure we are not missing anything.

* The `rwx` perms on i370/s390 mainframes came with a storage key,
  granting specific access to different memory regions. Do we have
  something similar here? Of course: `StorageNode`s are all distinct.

* The i370/s390 mainframes had a well-developed architecture of channels
  and subchannels. These are explicitly configured using `schib`s and
  `irb`s and whatnot. This resembles (even strongly?) the way in which
  the `ProxyNode`s are configured. That is, a bare-bones, ad-hoc
  structure is defined, at the same abstraction level as the thing
  carrying out the operations (`STARTIO`).

* Most other CPU complexes use PCIe as the networking fabric. This has
  concepts that include root ports, channels, devices. These are
  configured and controlled by PCI config registers at the hardware
  layer. This does not elicit any notably different or inspiring
  thoughts or suggestions for the design of the StorageNode.

* The actual devices hanging off the end of channels are disk drives,
  storage clusters, connected via fiber channel, ethernet, etc. The
  notable difference here is that topology configuration happens at
  a differrent software abstraction level then the hardware: The
  sysadmin gets a GUI dashboard with blinkenlights and knopkes.
  The programmer implementing the GUI gets a stack of libraries,
  each working at different abstraction layers. At this time, there
  is no comparable stack of interfaces for Atomese, so I don't extract
  any wisdom, here.

* The Ceph storage cluster offers several lessons. Formost is a negative
  lesson: file ownership and uid/gid management is outside the scope
  of CephFS. This is a holdover from Unix: uids and gids are numeric;
  the correlation to actual user accounts in `/etc/passwd` and
  `/etc/group` are decorellated. The correct long-term solution for
  Unix is to replace uid's & gid's by URLs, so as to provide a more
  global description of permissions.

* On the topic of permissions, we have the Nick Szabo e-rights system.
  The ability to accomplish something depended on having access to a
  crypto key for the particular action to be taken.

* On the topic of crypto, we have the idea of smart contracts, where
  multiple parties can engage in coopertive actions by authenticating
  with crypto keys.

How can we leverage the above for the next-gen design?
* The implementation of AtomSpace layers was challenging.
* The current implementation has an underwhelming performance profile.
* The current ProxyNode implementation is already version 2;
  the version 1 variant had proxies, but not ProxyNodes.

So we're effectively going for version 3, here; lets get it right,
instead of being half-assed about it.

### Redesigning layering
So what are the design requirements?

* We want "views", in that the current AtomSpace is authoritative in
  it's contents, hiding/modifying/updating contents that might be in
  other remote, attached AtomSpaces.
* So this is not really a "view", per se; it seems to be saying "get rid
  of the OverlayFS-like behavior, and provide it in some other way."
* `ProxyNodes` are passive, not active in thier update of AtomSpace
  contents. One has to explicitly fetch Atoms, Keys. This prevents the
  naive usage of Proxy infrastructure for the hiding/modifying behavior.
* Active updates is not the same as "push notifications". More on this
  later.

Layering has conflicting view/remove semantics. Lets review this:
* The "view" aspect is that everything visible in the current AtomSpace
  must actually be in the current AtomSpace, or is visible in a space
  below.
* Viewership could be accomplished by copy-in, except that this is
  wasteful when a lower layer is in the same RAM. Thus, COW was used.
* Remote proxy viewership has to be implemented as copy-in. This seems
  entirely doable with the current BackingStore API.
* The COW implementation also requires hiding, so that a deleted Atom
  is marked hidden.
* Local deletion does not create any difficulties when the proxy is
  remote.

The current layering implementation has COW in the current AtomSpace.
Can we push the COW into the proxy, instead?


### Are SensoryNode's AtomSpaces?
By logical extension of the above thoughts, SensoryNodes should be
AtomSpaces too ... I guess? But "remote" AtomSpaces are "remote",
whereas SensoryNodes are local interfaces to that "remoteness".

The inheritance hierarchy gets a bit messy; not all remote things
have to form of AtomSpaces ... or do they? It's always Values that
come back from remote sensing...



TODO List
---------
Writiing the above, the TODO list expands:
* The ThreadJoinLink should be removed from Atomese. DONE.
* Benchmark the Atompsace layers/frame performance. Maybe with a variant
  of the `LargeZipfUTest` layout, but this time with different count
  data at each layer.
