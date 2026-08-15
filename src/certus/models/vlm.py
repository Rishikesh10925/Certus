"""Frozen VLM backbone wrapper (LoRA weights, eval mode, no gradient) used
for batch inference over images to produce pooled visual features and/or
generated defect assertions. This is the one and only place inference on
the underlying vision-language model happens -- every nonconformity score
reads from here rather than re-running the model. See roadmap Section 3.1.
"""
