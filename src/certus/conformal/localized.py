"""Localized conformal prediction: weights calibration points by similarity
in the VLM's latent space via a nonlinear transform of cosine similarity.
Only attempt once a domain classifier confirms the backbone's
representation genuinely separates sites (t-SNE + AUC check, roadmap
Section 4.4). Sensitive to the similarity transform; the most
implementation-risky of the four shift-correction methods (Section 3.2.4).
"""
