"""THE corrected conformal quantile: ceil((n+1)(1-alpha)) / n, not 1-alpha.
This is the single most common conformal implementation bug in the wild
(roadmap Section 3.2.1) -- with small n the difference is the gap between a
valid and an invalid guarantee. Every other conformal module calls through
this one function; it is covered by tests/test_quantile.py and must never
be reimplemented inline elsewhere.
"""
