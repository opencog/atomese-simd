;
; vect-mult.scm
;
; Basic vector multiplication demo.

(use-modules (opencog) (opencog exec))
(use-modules (opencog sensory) (opencog opencl))

; View debug messages
(use-modules (opencog logger))
(cog-logger-set-stdout! #t)


; To run demo, copy `vec-mult.cl` in this directory to /tmp
; or alter the URL below.
;
(cog-execute!
	(Open
		(Type 'OpenclStream)
		(SensoryNode "opencl://Clover:AMD/tmp/vec-mult.cl")))
