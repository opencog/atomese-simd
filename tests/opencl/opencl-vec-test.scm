;
; opencl-vec-test.scm
;
; Basic OpenCL vector multiplication unit test.
;
(use-modules (srfi srfi-1))
(use-modules (opencog) (opencog exec))
(use-modules (opencog sensory) (opencog opencl))
(use-modules (opencog test-runner))

; (define clurl "opencl://Clover:AMD Radeon/tmp/vec-kernel.cl")
; (define clurl "opencl://CUDA:NVIDIA RTX 4000/tmp/vec-kernel.cl")

; Horrible hacker to extract location of the unit test.
(format #t "Command line: ~A\n" (command-line))
(define last-arg (last (command-line)))
(define pathcomp (string-split last-arg #\/))
(define leader (take pathcomp (- (length pathcomp) 1)))
(define path (fold
	(lambda (s t) (string-concatenate (list t "/" s)))
	"" leader))
(format #t "Unit test location: ~A\n" path)

; If unit test is run from cmake, then pathcomp is a long list.
; Otherwise unit test is run by hand, and its just cwd.
(define curloc (if (< 1 (length pathcomp)) path (getcwd)))

(define clurl (string-concatenate (list
	"opencl://:/" curloc "/vec-kernel.cl")))

(format #t "Looking for kernel at ~A\n" clurl)

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
(test-assert "mult one" (equal? (Number 2 4 6 8 10) kern-m1))

(define kern-m2 (cog-value-ref (cog-execute! gpu-location) 0))
(test-assert "mult two" (equal? (Number 2 4 6 8 10) kern-m2))

; ---------------------------------------------------------------
; Run it again, different data
(define krun-2
	(Write gpu-location
		(List
			(Predicate "vec_mult") ; Must be name of kernel
			(Number 1 2 3 4 5 6 7 8 9 10 11)
			(Number 2 3 4 5 6 5 4 3 2 1 0))))

(define kern-m3 (cog-execute! krun-2))
(test-assert "mult three"
	(equal? (Number 2 6 12 20 30 30 28 24 18 10 0) kern-m3))

(define kern-m4 (cog-value-ref (cog-execute! gpu-location) 0))
(test-assert "mult four"
	(equal? (Number 2 6 12 20 30 30 28 24 18 10 0) kern-m4))

; ---------------------------------------------------------------
; Run it again, with a different kernel
(define krun-3
	(Write gpu-location
		(List
			(Predicate "vec_add") ; Must be name of kernel
			(Number 1 2 3 4 5 6 7 8 9 10 11)
			(Number 2 3 4 5 6 5 4 3 2 1 0))))

(define kern-m5 (cog-execute! krun-3))
(test-assert "mult five"
	(equal? (Number 3 5 7 9 11 11 11 11 11 11 11) kern-m5))

(define kern-m6 (cog-value-ref (cog-execute! gpu-location) 0))
(test-assert "mult six"
	(equal? (Number 3 5 7 9 11 11 11 11 11 11 11) kern-m6))

; ---------------------------------------------------------------
; Use FloatValues
;
(cog-set-value!
	(Anchor "some data") (Predicate "some stream")
	(LinkValue
		(Predicate "vec_add")
		(FloatValue 0 0 0 0 0 0 0 0 0 0 0 0)
		(RandomStream 3)))

(define vector-stream
	(Write gpu-location
		(ValueOf (Anchor "some data") (Predicate "some stream"))))

(define kern-m7 (cog-execute! vector-stream))
(define kern-m8 (cog-value-ref (cog-execute! gpu-location) 0))
(test-assert "ran value" (equal? kern-m7 kern-m8))
(test-assert "ran type" (equal? 'FloatValue (cog-type kern-m7)))
(test-assert "ran size" (equal? 3 (length (cog-value->list kern-m7))))

; Run it again ... get different values
(define kern-m9 (cog-execute! vector-stream))
(define kern-m10 (cog-value-ref (cog-execute! gpu-location) 0))
(test-assert "ran change" (not (equal? kern-m7 kern-m9)))
(test-assert "ran2 value" (equal? kern-m9 kern-m10))
(test-assert "ran2 type" (equal? 'FloatValue (cog-type kern-m9)))
(test-assert "ran2 size" (equal? 3 (length (cog-value->list kern-m9))))

; ---------------------------------------------------------------
; Initialize the accumulator
(define vec-size 130)

(cog-set-value!
	(Anchor "some data") (Predicate "accumulator")
	(FloatValue (make-list vec-size 0)))

(define accum-location
	(ValueOf (Anchor "some data") (Predicate "accumulator")))

(cog-set-value!
	(Anchor "some data") (Predicate "accum task")
	(LinkValue
		(Predicate "vec_add") accum-location (RandomStream vec-size)))

; Define a feedback loop.
(define accum-stream
	(SetValue
		(Anchor "some data") (Predicate "accumulator")
		(Write gpu-location
			(ValueOf (Anchor "some data") (Predicate "accum task")))))

; Run it once ...
(define acc1 (cog-execute! accum-stream))
(test-assert "acc1 type" (equal? 'FloatValue (cog-type acc1)))
(test-assert "acc1 size" (equal? vec-size (length (cog-value->list acc1))))

; Run it lots ...
(define run-len 5123)
(for-each (lambda (x) (cog-execute! accum-stream)) (iota run-len 0))

(define accn (cog-execute! accum-location))
(test-assert "accn type" (equal? 'FloatValue (cog-type accn)))
(test-assert "accn size" (equal? vec-size (length (cog-value->list accn))))

; Result of repeated executation should be a large number,
; approx equal to 0.5 of vec-size * run-len by the central limit theorem
; and with stddev of sqrt of num samples.
(define vsum (fold + 0 (cog-value->list accn)))
(define vlen (* vec-size run-len))
(define vmean (/ vsum vlen))
(define vsigma (/ 1 (sqrt vlen)))

(format #t "sum: ~A len: ~A mean: ~A sigma: ~A\n" vsum vlen vmean vsigma)

; Acceptable deviation
(define accdev (* 5 vsigma))

(test-assert "accn lo bound" (< (- 0.5 accdev) vmean))
(test-assert "accn hi bound" (> (+ 0.5 accdev) vmean))

(test-end tname)
(opencog-test-end)
