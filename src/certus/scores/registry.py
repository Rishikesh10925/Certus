"""Nonconformity score registry. Every score (S1-S7, roadmap Section 3.3) is
registered here behind a single `@register('s3')` decorator interface so any
score can be swapped by config alone. S7 (the learned combination) must be
fitted strictly on a tuning split disjoint from calibration -- enforce that
separation here in code, not by discipline (Section 3.3, "three splits, not
two").
"""
