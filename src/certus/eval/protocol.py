"""The 100-seed evaluation harness: random calibration/test partitions
within each protocol, bootstrap confidence intervals (clustered by image
where the unit of analysis is a region), and seed logging so every
coverage number in the paper is reported as mean +/- SD rather than a
single lucky draw. See roadmap Section 5.5.
"""
