"""Small trainable heads attached to the frozen VLM backbone: a linear
classification head (feeds S1, roadmap Section 3.3) and an MC-dropout
wrapper for stochastic forward passes (feeds S5). Fit only on the
train/tune split -- never on calibration data.
"""
