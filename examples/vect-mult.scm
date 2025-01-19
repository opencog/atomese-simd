;
; vect-mult.scm
;
; Basic vector multiplication demo.

(use-modules (opencog) (opencog exec))
(use-modules (opencog sensory) (opencog opencl))

; View debug messages
(use-modules (opencog logger))
(cog-logger-set-stdout! #t)


(cog-execute!
	(Open
		(Type 'OpenclStream)
		(SensoryNode "opencl://Clover:AMD/foo.cl")))
