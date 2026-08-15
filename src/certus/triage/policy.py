"""Turns calibrated thresholds (lambda_hat, q_hat per site cluster) into a
three-outcome triage decision -- AUTO_CLEAR, FLAG, or ESCALATE -- rather
than a binary auto/human split, so a confident single-defect prediction
reaches a human with the defect type pre-populated instead of joining an
undifferentiated queue. See roadmap Section 3.4 for the reference
implementation.
"""
