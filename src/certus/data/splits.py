"""Generates and freezes the train/calibration/test split plus the
leave-one-site-out (LOSO) folds (roadmap Section 4.4). Splits are written
once to a versioned, content-hashed manifest on disk and every experiment
reads that manifest -- a split silently regenerating between runs is the
class of bug that produces months of uninterpretable results. The
calibration split must be physically inaccessible to any function whose
name begins with `fit_`.
"""
