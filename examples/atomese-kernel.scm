;
; atomese-kernel.scm
;
; Basic OpenCL GPU vector multiplication demo.
;
; Uses the Atomese "sensory" style API to open a channel to the
; OpenCL processing unit, and then send kernels and data there.
;
; Before running the demo, copy `vec-kernel.cl` in this directory to
; the `/tmp` directory; alternately, change the URL below.
;
; To run the demo, say `guile -s atomese-kernel.scm`. Alternately,
; cut-n-paste from this file to a guile command line.
;
; The demo has four parts:
; * Creating and opening a channel to the OpenCL device.
; * Sending a kernel along with NumberNode vector data the device.
; * Sending FloatValue data. Unlike NumberNodes, FloatValues are NOT
;   stored in the AtomSpace. This means they don't take up storage
;   space. This also means they are a bit harder to use, since they're
;   ephemeral, and disappear if not attached to an anchor point.
; * A demo of a feedback loop, implementing an accumulator.
;
(use-modules (opencog) (opencog exec))
(use-modules (opencog sensory) (opencog opencl))

; Optional but recommended; view debug messages.
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
; (define clurl "opencl://CUDA:NVIDIA RTX 4000/tmp/vec-kernel.cl")
(define clurl "opencl://:/tmp/vec-kernel.cl")

; Copy the kernel from here to /tmp
(copy-file "vec-kernel.cl" "/tmp/vec-kernel.cl")

; ---------------------------------------------------------------
; Define and open the device.
(define clnode (OpenclNode clurl))

; The argument to the *-open-* message is the type of the results
; that we want to receive. Two choices are avalabile: NumberNode
; and FloatValue.
(cog-execute!
	(SetValue clnode (Predicate "*-open-*") (Type 'FloatValue)))

; ---------------------------------------------------------------
; Now that it's open, define a simple stream that will write the name
; of a kernel and some vector data to the GPU/machine. This just defines
; what to do; nothing is done until this is executed.
(define kernel-runner
	(SetValue clnode (Predicate "*-write-*")
		(Section
			(OpenclKernel "vec_mult") ; Must be name of kernel
			(ConnectorSeq
				(Type 'Number)
				(Number 1 2 3 4 5)
				(Number 2 2 2 2 2 2 3 42 999)
				(Connector (Number -1))))))

; Run the kernel.
(cog-execute! kernel-runner)

; Get the result.
(cog-execute! (ValueOf clnode (Predicate "*-read-*")))

; ---------------------------------------------------------------
; Run it again, with different data.
(cog-execute!
	(SetValue clnode (Predicate "*-write-*")
		(Section
			(OpenclKernel "vec_mult") ; Must be name of kernel
			(ConnectorSeq
				(Type 'Number)
				(Number 1 2 3 4 5 6 7 8 9 10 11)
				(Number 2 3 4 5 6 5 4 3 2 1 0)))))

; Get the result.
(cog-execute! (ValueOf clnode (Predicate "*-read-*")))

; ---------------------------------------------------------------
; Run it again, with a different kernel (addition this time, not
; multiplication.)
(cog-execute!
	(SetValue clnode (Predicate "*-write-*")
		(Section
			(OpenclKernel "vec_add") ; Must be name of kernel
			(ConnectorSeq
				(Type 'Number)
				(Number 1 2 3 4 5 6 7 8 9 10 11)
				(Number 2 3 4 5 6 5 4 3 2 1 0))))

; Get the result
(cog-execute! (ValueOf clnode (Predicate "*-read-*")))

; ---------------------------------------------------------------
; Instead of using NumberNodes, use FloatValues.
; Both types can hold vectors of floats. NumberNodes are stored in
; the AtomSpace (and thus clog things up), while FloatValues are not.
; Which is great, as usually there are lots of them.
; The trade-off is that the Values have to be put somewhere where they
; can be found. i.e. anchored "some where".
;
; (RandomStream N) creates a vector of N random numbers. These numbers
; change with every access (which is why it is called a "stream" instead
; of a "vector".)
;
(cog-set-value!
	(Anchor "some data") (Predicate "some stream")
	(SectionValue
		(OpenclKernel "vec_add")
		LinkValue
			(Type 'FloatValue)
			(FloatValue 0 0 0 0 0 0 0 0 0 0 0 0)
			(RandomStream 3))))

; Define Atomse that will send data to GPUs.
(define vector-stream
	(SetValue clnode (Predicate "*-write-*")
		(ValueOf (Anchor "some data") (Predicate "some stream"))))

; Run it once ...
(cog-execute! vector-stream)
(cog-execute! (ValueOf clnode (Predicate "*-read-*")))

; Run it again ...
(cog-execute! vector-stream)
(cog-execute! (ValueOf clnode (Predicate "*-read-*")))

; --------- The End! That's All, Folks! --------------
