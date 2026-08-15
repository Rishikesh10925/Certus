"""Learn-then-Test -- converts risk control into a multiple-hypothesis-
testing problem over a finite lambda grid, using Hoeffding-Bentkus
p-values and fixed-sequence FWER control, giving
P(R(lambda_hat) <= epsilon) >= 1 - delta. The high-probability,
regulator-grade statement. See roadmap Section 3.2.3.
"""
