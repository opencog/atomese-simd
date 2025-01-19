;
; atomese-kernel.scm
;
; Basic OpenGL GPU vector multiplication demo.
; Uses the Atomese "sensory" style API to open a channel to the
; OpenCL processing unit, and send data there.
;
; Before running the demo, copy `vec-kernel.cl` in this directory to
; the `/tmp` directory, or alter the URL below.
;
; To run the demo, say `guile -s atomese-kernel.scm`
;
(use-modules (opencog) (opencog exec))
(use-modules (opencog sensory) (opencog opencl))

; Optional; view debug messages
(use-modules (opencog logger))
(cog-logger-set-stdout! #t)

; URL specifying platform, device and program source.
; Modify as needed. Any substrings will do for platform and device,
; including empty strings. The first match found will be used.
; If in doubt, try 'opencl://:/tmp/vec-kernel.cl`
;
; Must be of the form 'opencl://platform/device/path/to/kernel.cl'
; Can also be clcpp or spv file.
; (define clurl "opencl://Clover:AMD Radeon/tmp/vec-kernel.cl")
(define clurl "opencl://:/tmp/vec-kernel.cl")

; ---------------------------------------------------------------
; Brute-force open. This checks the open function works.
; Optional; don't need to do this, except to manually check
; things out.
(cog-execute!
	(Open
		(Type 'OpenclStream)
		(SensoryNode clurl)))

; ---------------------------------------------------------------
; Define some Atomese to open connection, and place it where can
; be found later.
(define do-open-device
	(SetValue
		(Anchor "some gpus") (Predicate "some gpu channel")
		(Open
			(Type 'OpenclStream)
			(SensoryNode clurl))))

; Go ahead and open it.
(cog-execute! do-open-device)

; ---------------------------------------------------------------
; Now that it's open, define a simple stream that will write the name
; of a kernel and some vector data to the GPU/machine. This just defines
; what to do; nothing is done until this is executed.
(define kernel-runner
	(Write
		(ValueOf (Anchor "some gpus") (Predicate "some gpu channel"))
		(List
			(Predicate "vec_mult") ; Must be name of kernel
			(Number 1 2 3 4 5)
			(Number 2 2 2 2 2 2 3 42 999))))

; Run the kernel.
(cog-execute! kernel-runner)

; Get the result
(format #t "Result from running kernel is ~A\n"
	(cog-execute!
		(ValueOf (Anchor "some gpus") (Predicate "some gpu channel"))))

; Do it again ... Nothing changed.
(format #t "Once again ...its ~A\n"
	(cog-execute!
		(ValueOf (Anchor "some gpus") (Predicate "some gpu channel"))))

; ---------------------------------------------------------------
; Run it again, with different data.
(cog-execute!
	(Write
		(ValueOf (Anchor "some gpus") (Predicate "some gpu channel"))
		(List
			(Predicate "vec_mult") ; Must be name of kernel
			(Number 1 2 3 4 5 6 7 8 9 10 11)
			(Number 2 3 4 5 6 5 4 3 2 1 0))))

; Get the result
(format #t "And now, with different data ... ~A\n"
	(cog-execute!
		(ValueOf (Anchor "some gpus") (Predicate "some gpu channel"))))

; ---------------------------------------------------------------
; Run it again, with a different kernel (addition this time, not
; multiplication.)
(cog-execute!
	(Write
		(ValueOf (Anchor "some gpus") (Predicate "some gpu channel"))
		(List
			(Predicate "vec_add") ; Must be name of kernel
			(Number 1 2 3 4 5 6 7 8 9 10 11)
			(Number 2 3 4 5 6 5 4 3 2 1 0))))

; Get the result
(format #t "Addding, instead of multiplying ... ~A\n"
	(cog-execute!
		(ValueOf (Anchor "some gpus") (Predicate "some gpu channel"))))

; ---------------------------------------------------------------
; Instead of using NumberNodes, use FloatValues.
(cog-set-value!
	(Anchor "some data") (Predicate "some stream")
	(LinkValue
		(Predicate "vec_add")
		(FloatValue 0 0 0 0 0 0 0 0)
		(FloatValue 1 2 3 4 5)))

(cog-execute!
	(Write
		(ValueOf (Anchor "some gpus") (Predicate "some gpu channel"))
		(ValueOf (Anchor "some data") (Predicate "some stream"))))

(format #t "Float stream results ... ~A\n"
	(cog-execute!
		(ValueOf (Anchor "some gpus") (Predicate "some gpu channel"))))

; --------- The End! That's All, Folks! --------------
