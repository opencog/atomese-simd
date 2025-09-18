Design Notes F
==============
September 2025

Notes about implementing Atomese descriptions of RNN's an LSTM.

Describing NN's with Atomese
----------------------------
The task here is to be able to describe the structure of RNN's, LSTM and
similar networks in "pure Atomese", i.e. using PlusLink, TimesLink and
so on to describe the vector operations.

Embedding Atoms into RNN and LSTM
----------------------------------
A subtrask is to create vector embeddings for Atoms.  Several types of
vector embeddings are posible:
* Every Atom has a 64-bit hash value; this could be converted to a
  BoolValue and treated as a bit-vector embedding. These are currently
  almost entirely collision-free.
* 256-bit hashes are also reasonable.
* Conventional RNN's and LSTM's use float vector embeddings. How would this
  work?

Vector Embeddings
-----------------
A reasonable way of doing a vector embedding for some Atom is to have
```
    (cog-set-value! (Atom...) (Predicate "some-key") (FloatValue...))
```
where the `FloatValue` would be N floats long (for N=768 or 1024 or
whatever), and the initial value would be generated using a deterministic
algo that used the 64-bit atom hash value as an RNG seed, generating N
floats (deterministically) from this seed.

Several issues:
* What should they key be?
* How is the float vector generated?
* What is being predicted/trained on?

### What should they key be?
We need to allow for (many) different embeddings, all available at the same
time. The specific set of Atoms to be embedded in some specific network
emerge as the result of running some QueryLink over the AtomSpace. This is
similar to how the "matrix" code works today: the vocabulary is limited to
a specific set of Atoms that respond to some specific Query (or other
identification.)

Thus, it makes sense for the key to be the specific Atom, usually a
QueryLink, but also maybe any other atom that specifies the specific
vocabulary for the specific embedding.

The base assumption here is that, during training, the embedding vector
will change; thus, we cannot have one global unique embedding vector per
Atom, as it is something that will be changing. Thus, it is part of the
specific model.

### Initial Embedding API
An initial embedding that is a deterministic function of the Atom hash
value seems entirely reasonable. What is the corret API for producing
this? It needs to have:

* Length: 512, 768, 1024 or more elts.
* Type: FloatValue (for doubles), Float32Value, Float16Value, BoolValue

The olde-school technique would be to write
```
	(cog-execute!
		(VectorEmbedLink
			(Atom ...)
			(NumberNode 768)
			(TypeNode 'Float32Value)))
```
which would return the desired initial embedding. This works. Some
problems:
* Requires actually creating the Link. But this link is not really
  needed for any kind of long-term use/re-use.
* But this is not really a problem, after all. Just use `ValueOf` or
  `PureExec` to pipe Atoms into this. See hand-wringing agonizing below.

A new-style OO API could be
```
	(cog-set-value!
		(EmbedNode "vector://Float32Value/768")
		(Predicate "*-write-*)
		(ListValue
		   (Atom ...)
         (KeyWhereAtom ...)
			(NumberNode 768)
			(TypeNode 'Float32Value)))
```
But this is terrible, because the URL is ugly.
* Object URL's are supposed to represent external locations, but this
  doesn't really need to be external.
* URL's imply the need for string manipulation, and the AtomSpace Node
  API is not string oriented.
* Someday, Node names should be vectors, but that day has not yet arrived.
* There is no need for an open-close-read-write API, so having the OO
  API does not seem to be needed.
* There's no reason to have a URL here, anyway...

Overall: the OO API seems pointless and useless.

### Stateless Executation
A meta-problem might be that we needs something in between `cog-execute!`
and `cog-set-value!`.  The problem with `cog-execute!` is it takes just
one argument: the Atom to execute.  Thus, a Link must be created to wrap
arguments (e.g. `ExecutionOutputLink` as the prototypical, canonical
example.)  The `cog-set-value!` avoids this, as it specifies keys and
arguments. The problem with it is that it is completely async: it has no
return value; useing `cog-value` to read the result forces it to be
stateful.  Thus, a completely stateless intermediate form between
`cog-execute!` and `cog-set-value!` is desirable. Or so it would seem.

The backwards compat solution would be to change `Atom::setValue()` to
not return void, and so on down the line. So `(cog-exeucte! (SetValue...))`
would then return not VoidValue, but something meaningful...

It seems like having this might make it easier to specify flows. For
example, this would allow us to ditch things like `ExecutionOutputLink`.
But is anything really gained, here? Analysis of the flow of control
seems to get ... more complicated. Hmm. All this is very unclear.

Conclusion: This is some kind og agony hand-wringing. We already have
everything needed with things like `ValueOf` link, and the executable
atoms that return `QueueValue` and whatnot. We can just pipe things.
So the answer is: don't change anything. What we have already is just
fine...

### Embedding Flow.

Another "fix" for the flow situation is to write
```
	(cog-execute!
		(VectorEmbedLink
			(ValueOf (Anchor "somewhere") (Predicate "some key"))
			(NumberNode 768)
			(TypeNode 'Float32Value)))
```
and so the generation of the initial embeddings becomes a flow.
If its a query, then
```
	(cog-execute!
		(VectorEmbedLink
			(PureExec (QueryLink ...))
			(NumberNode 768)
			(TypeNode 'Float32Value)))
```

Then write to where the embedding will be stored:
```
	(cog-execute!
		(SetValueLink
			(Anchor "some place") (Predicate "some embedding key")
			(VectorEmbedLink ...)))
```
