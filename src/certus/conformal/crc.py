"""Conformal risk control (Angelopoulos et al., ICLR 2024) -- the primary
method. Bounds E[L_test(lambda_hat)] <= epsilon for the monotone,
asymmetric missed-defect loss rather than symmetric coverage. Monotonicity
of L(lambda) must be verified explicitly (a non-increasing check) -- a sign
error here silently returns garbage. See roadmap Section 3.2.2.
"""
