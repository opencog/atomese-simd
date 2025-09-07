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

(define clnode (OpenclNode clurl))
(format #t "testing ~A" clnode)
(cog-execute!
   (SetValue clnode (Predicate "*-open-*") (Type 'FloatValue)))

; Test to see if stream is open.
(define cnct (cog-execute! (ValueOf clnode (Predicate "*-connected?-*"))))
(format #t "connected? ~A" cnct)
(test-assert "open stream" (cog-value-ref cnct 0))

; ---------------------------------------------------------------
(define kernel-runner
	(SetValue clnode (Predicate "*-write-*")
		(List
			(Predicate "vec_mult") ; Must be name of kernel
			(Number 1 2 3 4 5)
			(Number 2 2 2 2 2 2 3 42 999))))

(cog-execute! kernel-runner)
(define kern-m1
	(cog-execute! (ValueOf clnode (Predicate "*-read-*"))))
(test-assert "mult one" (equal? (FloatValue 2 4 6 8 10) kern-m1))

; ---------------------------------------------------------------
; Run it again, different data
(define krun-2
	(SetValue clnode (Predicate "*-write-*")
		(List
			(Predicate "vec_mult") ; Must be name of kernel
			(Number 1 2 3 4 5 6 7 8 9 10 11)
			(Number 2 3 4 5 6 5 4 3 2 1 0))))

(cog-execute! krun-2)
(define kern-m3
	(cog-execute! (ValueOf clnode (Predicate "*-read-*"))))
(test-assert "mult three"
	(equal? (FloatValue 2 6 12 20 30 30 28 24 18 10 0) kern-m3))

; ---------------------------------------------------------------
; Run it again, with a different kernel
(define krun-3
	(SetValue clnode (Predicate "*-write-*")
		(List
			(Predicate "vec_add") ; Must be name of kernel
			(Number 1 2 3 4 5 6 7 8 9 10 11)
			(Number 2 3 4 5 6 5 4 3 2 1 0))))

(cog-execute! krun-3)
(define kern-m5
	(cog-execute! (ValueOf clnode (Predicate "*-read-*"))))
(test-assert "mult five"
	(equal? (FloatValue 3 5 7 9 11 11 11 11 11 11 11) kern-m5))

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
(define run-kernel
	(SetValue clnode (Predicate "*-write-*")
		(ValueOf (Anchor "some data") (Predicate "accum task"))))

(define update-data
	(SetValue
		(Anchor "some data") (Predicate "accumulator")
		(ValueOf clnode (Predicate "*-read-*"))))

; Run it once ...
(cog-execute! run-kernel)
(define acc1 (cog-execute! update-data))
(test-assert "acc1 type" (cog-subtype? 'FloatValue (cog-type acc1)))
(test-assert "acc1 size" (equal? vec-size (length (cog-value->list acc1))))

; Run it lots ...
(define run-len 5123)
(for-each
	(lambda (x)
		(cog-execute! run-kernel)
		(cog-execute! update-data))
	(iota run-len 0))

(cog-execute! run-kernel)
(define accn (cog-execute!  update-data))
(test-assert "accn type" (cog-subtype? 'FloatValue (cog-type accn)))
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
