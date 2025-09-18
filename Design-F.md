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

