Design Notes F
==============
September 2025

Notes about implementing Atomese descriptions of RNN's an LSTM.

Embedding Atoms into RNN and LSTM
----------------------------------
The task here is to be able to describe the structure of RNN's, LSTM and
similar networks in "pure Atomese", i.e. using PlusLink, TimesLink and
so on to describe the vector operations.

A subtrask is to create vector embeddings for Atoms.  Several types of
vector embeddings are posible:
* Every Atom has a 64-bit hash value; this could be converted to a
  BoolValue and treated as a bit-vector embedding. These are currently
  almost entirely collision-free.
* 256-bit hashes are also reasonable.
* Conventional RNN's and LSTM's use float vector embeddings. How would this
  work?

