;
; accumulator.scm
;
; Demo of an accumulator. Demonstrates how to designate a location in
; the AtomSpace (i.e. in system RAM) that can hold vector data that
; shadows what the GPU is working with, while avoiding un-nneded copying
; of that data between the system and the GPU.
;
(use-modules (opencog) (opencog exec))
(use-modules (opencog sensory) (opencog opencl))

; Optional but recommended; view debug messages.
(use-modules (opencog logger))
(cog-logger-set-stdout! #t)

; URL specifying platform, device and program source.
; See `atomese-kernel.scm` for move about this.
(define clurl "opencl://:/tmp/vec-kernel.cl")

; ---------------------------------------------------------------
; Define and open the device.
(define clnode (OpenclNode clurl))

(cog-execute!
	(SetValue clnode (Predicate "*-open-*") (Type 'FloatValue)))

; ---------------------------------------------------------------
; The code below defines an accumualator, and uses the GPU to add
; vectors into it. The accumulator stays of the GPU, until it is
; examined, at which point it's current value is downloaded to system
; memory. Printing the vector counts as "examining" it; it's latest
; value will be downloaded automatically.
;
; The accumualator is implemented as an OpenclFloatValue
; (a specialization of the ordinary FloatValue) attached to a specific
; key on a specific Atom: a designated "well-known location".
;
; The demo runs the "vec_add" GPU kernel in a loop, adding new random
; vectors into the accumulator. The accumulator is specified as both
; input and output to the "vec_add" kernel: that's why it accumulates.
;
; The random vectors are generated in system memory, and are uploaded
; to the GPU each cycle. The upload is done automatically every time the
; kernel is executed.  The accumulator stays in the GPU, and is not
; moved around (unless, of course, it is "examined" as described above.)
;
; -------
; Initialize the accumulator to all-zeros, and anchor it where
; it can be found.
(cog-set-value!
	(Anchor "some place") (Predicate "accumulator")
	(OpenclFloatValue 0 0 0 0 0))

; When executed, this will return the current accumulator value.
(define accum-location
	(ValueOf (Anchor "some place") (Predicate "accumulator")))
(cog-execute! accum-location)

; Send the zeroed-out accumulator up to the GPU. It needs to be
; there, as it will be read from the very first execution of the kernel.
(cog-set-value! clnode (Predicate "*-write-*") accum-location)
(cog-execute! (ValueOf clnode (Predicate "*-read-*")))

; Take a peek. See that it is all-zero, as expected.
(cog-execute! accum-location)

; -------
; Initialize a data source. In this case, a random number generator.
(cog-set-value!
   (Anchor "some place") (Predicate "data source")
   (RandomStream 5))

(define source-location
   (ValueOf (Anchor "some place") (Predicate "data source")))

; Take a peek. The random numbers will change each time we look. This
; is a streaming source, and we are sampling from the stream.
(cog-execute! source-location)
(cog-execute! source-location)
(cog-execute! source-location)

; -------
; Put it all together.  Bind the "vec_add" GPU kernel to the accumulator
; and the data source.
(define kerny
	(Section
		(Predicate "vec_add")
		(ConnectorSeq accum-location accum-location source-location)))

; Define a pair of functions. The first runs the kernel, defined above,
; and the second gets the status of that run. Printing the status will
; print the actual job that was run (an OpenclJobValue) and that print
; will cause an "examination" of the accumulator, and so the current
; value of the accumulator will be printed in the job.
(define run-kernel
	(SetValue clnode (Predicate "*-write-*") kerny))

(define get-status
   (ValueOf clnode (Predicate "*-read-*")))

; Run it once ...
(cog-execute! run-kernel)
(cog-execute! get-status)

; And again ...
(cog-execute! run-kernel)
(cog-execute! get-status)

; Put it in a loop.
(define (loopy N)
	(cog-execute! run-kernel)
	(cog-execute! get-status)
	(if (< 0 N) (loopy (- N 1))))

; Run the loop for a while. Although the status is fetched each time,
; it is not printed, and so the accumulator vector is not downloaded
; to the system.
(loopy 51)

; Take a look at the accumulator. Since each random number is uniformly
; distributed between 0.0 and 1.0, we expect the accumulator to
; accumulate about 0.5 on average, each run.
(cog-execute! accum-location)


; --------- The End! That's All, Folks! --------------
