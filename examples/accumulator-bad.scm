;
; accumulator-bad.scm
;
; "Bad" demo of an accumulator. The demo works but is a "bad" demo
; because it performs excessive and pointless data movement between
; the GPU and system memory.
;
; The redeeming aspect of this demo is it shows how to designate a
; location in the AtomSpace (i.e. in system RAM) that can hold vector
; data recently obtained from the GPU.
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
; vectors into it. The result of the addition is downloaded back into
; the AtomSpace.
;
; The accumualator is a FloatValue attached to a specific key on a
; specific Atom: a designated "well-known location". For each cycle,
; a new random vector is created. This new vector, together with the
; value in the accumulator, are uploaded to the GPU, where they are
; addec with the `vec_add` kernel. The result is then downloaded back
; into system RAM (into the AtomSpace), ad the "well known location".
;
; This is a "bad" demo, in that the accumulator value is repeated
; uploaded and downloaded every step. This is a waste of compute
; resources; it would be better to just keep the accumulator on the
; GPU up until when it is actually needed.
;
; However, as a demo, this works. It shows how to designate a location
; where a vector can be stored, and how to work with that vector.

; ----
; Initialize the accumulator to all-zeros, and anchor it where
; it can be found.
(cog-set-value!
	(Anchor "some place") (Predicate "accumulator")
	(OpenclFloatValue 0 0 0 0 0))

; When executed, this will return the current accumulator value.
(define accum-location
	(ValueOf (Anchor "some place") (Predicate "accumulator")))
(cog-execute! accum-location)

; Send the zeroed-out accumulator up to the GPU.
(cog-set-value! clnode (Predicate "*-write-*") accum-location)
(cog-execute! (ValueOf clnode (Predicate "*-read-*")))

; Bind the vector-add GPU kernel to a pair of input vectors. The first
; vector is the accumulator, and the second one is a vector of three
; random numbers. Note that these random numbers change with every
; invocation.
(cog-set-value!
	(Anchor "some place") (Predicate "accum task")
	(SectionValue
		(OpenclKernel clnode (Predicate "vec_add"))
		(LinkValue accum-location accum-location (RandomStream 3))))

; Define a pair of functions. The first runs the kernel, defined above,
; and the second downloads the results. Calling this pair in succession
; increments the accumulator with random values.
(define run-kernel
	(SetValue clnode (Predicate "*-write-*")
		(ValueOf (Anchor "some place") (Predicate "accum task"))))


; Run it once ...
(cog-execute! run-kernel)
(cog-execute! get-result)

; And again and again ...
(cog-execute! run-kernel)
(cog-execute! get-result)

(cog-execute! run-kernel)
(cog-execute! get-result)

(cog-execute! run-kernel)
(cog-execute! get-result)

; --------- The End! That's All, Folks! --------------
