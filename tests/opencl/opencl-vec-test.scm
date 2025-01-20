;
; opencl-vec-test.scm
;
; Basic OpenCL vector multiplication unit test.
;
(use-modules (opencog) (opencog exec))
(use-modules (opencog sensory) (opencog opencl))
(use-modules (opencog test-runner))

; (define clurl "opencl://Clover:AMD Radeon/tmp/vec-kernel.cl")
(define clurl "opencl://:/tmp/vec-kernel.cl")

(opencog-test-runner)
(define tname "opencl-vec-test")
(test-begin tname)

; ---------------------------------------------------------------

(define open-stream (cog-execute!
	(Open (Type 'OpenclStream) (SensoryNode clurl))))

; Crash if stream not corrrectly intialized.
(define stream-str (format #f "~A" open-stream))
(test-assert "open stream" (equal? "(OpenclStream)\n" stream-str))

(format #t "Opened stream is ~A\n" open-stream)

; ---------------------------------------------------------------
(define do-open-device
	(SetValue (Anchor "some gpus") (Predicate "some gpu channel")
		(Open
			(Type 'OpenclStream)
			(SensoryNode clurl))))

(define reopen-stream (cog-execute! do-open-device))
(define restream-str (format #f "~A" reopen-stream))
(test-assert "reopen stream" (equal? "(OpenclStream)\n" restream-str))

; Define some short-hand for the anchor-point.
(define gpu-location
	(ValueOf (Anchor "some gpus") (Predicate "some gpu channel")))

(define loc-stream (cog-execute! gpu-location))
(define loc-str (format #f "~A" loc-stream))
(test-assert "loc stream" (equal? "(OpenclStream)\n" loc-str))

; ---------------------------------------------------------------
(define kernel-runner
	(Write gpu-location
		(List
			(Predicate "vec_mult") ; Must be name of kernel
			(Number 1 2 3 4 5)
			(Number 2 2 2 2 2 2 3 42 999))))

(define kern-m1 (cog-execute! kernel-runner))

; Get the result
(format #t "Result from running kernel is ~A\n"
	(cog-execute! gpu-location))

; Do it again ... Nothing changed.
(format #t "Once again ...its ~A\n"
	(cog-execute! gpu-location))

; ---------------------------------------------------------------
; Run it again, with different data.
(cog-execute!
	(Write gpu-location
		(List
			(Predicate "vec_mult") ; Must be name of kernel
			(Number 1 2 3 4 5 6 7 8 9 10 11)
			(Number 2 3 4 5 6 5 4 3 2 1 0))))

; Get the result
(format #t "And now, with different data ... ~A\n"
	(cog-execute! gpu-location))

; ---------------------------------------------------------------
; Run it again, with a different kernel (addition this time, not
; multiplication.)
(cog-execute!
	(Write gpu-location
		(List
			(Predicate "vec_add") ; Must be name of kernel
			(Number 1 2 3 4 5 6 7 8 9 10 11)
			(Number 2 3 4 5 6 5 4 3 2 1 0))))

; Get the result
(format #t "Adding, instead of multiplying ... ~A\n"
	(cog-execute! gpu-location))

; ---------------------------------------------------------------
; Instead of using NumberNodes, use FloatValues.
; Both types can hold vectors of floats. NumberNodes are stored in
; the AtomSpace (and thus clog things up), while FloatValues are not.
; Which is great, as usually there are lots of them.
; The trade-off is that the Values have to be put somewhere where they
; can be found. i.e. anchired "some where".
;
; (RandomStream N) creates a vector of N random numbers. These numbers
; change with every access (which is why it is called a "stream" instead
; of a "vector".)
;
(cog-set-value!
	(Anchor "some data") (Predicate "some stream")
	(LinkValue
		(Predicate "vec_add")
		(FloatValue 0 0 0 0 0 0 0 0 0 0 0 0)
		(RandomStream 3)))

; Define Atomse that will send data to GPUs.
(define vector-stream
	(Write gpu-location
		(ValueOf (Anchor "some data") (Predicate "some stream"))))

; Run it once ...
(cog-execute! vector-stream)
(format #t "Random numbers from a stream ...\n~A\n"
	(cog-execute! gpu-location))

; Run it again ...
(cog-execute! vector-stream)
(format #t "More random numbers ...\n~A\n"
	(cog-execute! gpu-location))

; ---------------------------------------------------------------
; Similar to above, but feed back the results of addition into the
; same location. This implements an accumulator. The addition is
; performed on the GPU, and, with each iteration, the result is
; pulled out (back into system memory, into an Atomese FloatValue)
; which is then used for the next round.

; Initialize the accumulator to all-zeros, and anchor it where
; it can be found.
(cog-set-value!
	(Anchor "some data") (Predicate "accumulator")
	(FloatValue 0 0 0 0 0))

; When executed, this will return the current accumulator value.
(define accum-location
	(ValueOf (Anchor "some data") (Predicate "accumulator")))

; Bind the vector-add GPU kernel to a pair of input vectors. The first
; vector is the accumulator, and the second one is a vector of three
; random numbers. Note that these random numbers change with every
; invocation.
(cog-set-value!
	(Anchor "some data") (Predicate "accum task")
	(LinkValue
		(Predicate "vec_add") accum-location (RandomStream 3)))

; Define a feedback loop. With each invocation, the accumulator will
; be updated with the result of the addition.
(define accum-stream
	(SetValue
		(Anchor "some data") (Predicate "accumulator")
		(Write gpu-location
			(ValueOf (Anchor "some data") (Predicate "accum task")))))

; Run it once ...
(format #t "Accumulator stream results ...\n  ~A\n"
	(cog-execute! accum-stream))

(format #t "Again ...\n  ~A\n"
	(cog-execute! accum-stream))

(format #t "And again ...\n  ~A\n"
	(cog-execute! accum-stream))

(cog-execute! accum-stream)
(cog-execute! accum-stream)
(cog-execute! accum-stream)
(cog-execute! accum-stream)

(format #t "Many more times ...\n  ~A\n"
	(cog-execute! accum-stream))

(test-end tname)
(opencog-test-end)
