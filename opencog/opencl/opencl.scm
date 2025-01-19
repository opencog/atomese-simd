
;
; OpenCL atom-types module
;
(define-module (opencog opencl))

(use-modules (opencog))
(use-modules (opencog opencl-config))

; Load the C library that calls the nameserver to load the types.
(load-extension
   (string-append opencog-ext-path-opencl-types "libopencl-types")
   "opencl_types_init")

(load-extension
   (string-append opencog-ext-path-opencl "libopencl")
   "opencog_opencl_init")

(include-from-path "opencog/opencl/opencl_types.scm")
