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
`(PureExec AST)`

