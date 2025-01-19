;
; vect-mult.scm
;
; Basic OpenGL GPU vector multiplication demo.
; Uses the Atomese "sensory" style API to open a channel to the
; OpenCL processing unit, and send data there.
;
; Before running the demo, copy `vec-mult.cl` in this directory to
; the `/tmp` directory, or alter the URL below.
;
; To run the demo, say `guile -s vect-mult.scm`
;
(use-modules (opencog) (opencog exec))
(use-modules (opencog sensory) (opencog opencl))

; Optional; view debug messages
(use-modules (opencog logger))
(cog-logger-set-stdout! #t)

; URL specifying platform, device and program source.
; Modify as needed. Any substrings will do for platform and device,
; including empty strings. The first match found will be used.
; If in doubt, try 'opencl://:/tmp/vec-mult.c`
;
; Must be of the form 'opencl://platform/device/path/to/kernel.cl'
; Can also be clcpp or spv file.
; (define clurl "opencl://Clover:AMD Radeon/tmp/vec-mult.cl")
(define clurl "opencl://:/tmp/vec-mult.cl")

; Brute-force open. This checks the basic functions.
(cog-execute!
	(Open
		(Type 'OpenclStream)
		(SensoryNode clurl)))

; Atomese to open connection and place it where we can find it.
(define do-open-device
	(SetValue
		(Anchor "some gpus") (Predicate "gpu channel")
		(Open
			(Type 'OpenclStream)
			(SensoryNode clurl))))

; Go ahead and open it.
(cog-execute! do-open-device)

; Now that it's open, define a simple stream that will write the name
; of a kernel and some vector data to the GPU/machine. This just defines
; what to do; nothing is done until this is executed.
(define do-mult-vecs
	(Write
		(ValueOf (Anchor "some gpus") (Predicate "gpu channel"))
		(List
			(Predicate "vec_mult") ; Must be name of kernel
			(Number 1 2 3 4 5)
			(Number 2 2 2 2 2 2 3 42 999))))

; Perform the actual multiply
(cog-execute! do-mult-vecs)

; Get the result
(format #t "Result from running kernel is ~A\n"
	(cog-execute!
		(ValueOf (Anchor "some gpus") (Predicate "gpu channel"))))

; Do it again ...
(format #t "Once again ...its ~A\n"
	(cog-execute!
		(ValueOf (Anchor "some gpus") (Predicate "gpu channel"))))
