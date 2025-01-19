;
; vect-mult.scm
;
; Basic vector multiplication demo.

(use-modules (opencog) (opencog exec))
(use-modules (opencog sensory) (opencog opencl))

(cog-execute!
	(Open
		(Type 'OpenclStream)
		(SensoryNode "opencl:///AMD/foo.cl")))
