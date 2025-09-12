;
; dot-product-bad.scm
;
; An example of taking dot products between a pair of vectors.
; Its "bad" in that this example does not quite work yet, at least,
; not in the way in which it's desired to work.
;
(use-modules (opencog) (opencog exec))

(cog-set-value! (Anchor "location") (Predicate "vetor-pairs")
	(LinkValue
		(FloatValue 1 2 3 4 5)
		(FloatValue 1 1 1 2 2)))

(define pair-location
	(FloatValueOf (Anchor "location") (Predicate "vetor-pairs")))

(cog-execute! (Plus pair-location))
