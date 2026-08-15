"""Weighted conformal prediction under covariate shift (Tibshirani et al.
2019): reweights calibration points by an estimated likelihood ratio
w(x) = p_test(x) / p_cal(x) from a domain classifier over VLM features.
Valid only when P(Y|X) is stable across the shift. Must log the effective
sample size every run -- it collapses when weights are extreme and is the
diagnostic that flags silent failure. See roadmap Section 3.2.4.
"""
