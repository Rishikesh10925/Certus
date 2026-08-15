"""Split conformal prediction: LAC (Sadinle et al.), APS (Romano et al.),
and RAPS (Angelopoulos et al.) -- the three non-risk-control conformal
baselines (B5-B7). Guarantee: P(Y_test in C(X_test)) >= 1 - alpha, for
exchangeable data, any model, any n. See roadmap Section 3.2.1.
"""
