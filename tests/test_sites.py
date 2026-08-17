"""Covers certus.data.sites -- the clustering and domain-classifier
diagnostic that decides whether proxy sites are trustworthy enough for
LOSO. The load-bearing property under test: well-separated groups score a
high AUC, and a single homogeneous population scores close to chance."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs

from certus.data.sites import choose_k, cluster_sites, domain_classifier_auc


def test_domain_classifier_auc_is_high_for_separated_clusters() -> None:
    features, true_labels = make_blobs(
        n_samples=300, centers=3, n_features=8, cluster_std=0.5, random_state=0
    )
    auc = domain_classifier_auc(features.astype(np.float32), true_labels, seed=0)
    assert auc > 0.95


def test_domain_classifier_auc_is_near_chance_for_random_labels() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(300, 8)).astype(np.float32)
    labels = rng.integers(0, 3, size=300)
    auc = domain_classifier_auc(features, labels, seed=0)
    assert 0.35 < auc < 0.65


def test_cluster_sites_recovers_known_cluster_count() -> None:
    features, true_labels = make_blobs(
        n_samples=300, centers=4, n_features=8, cluster_std=0.3, random_state=0
    )
    k_scores = choose_k(features.astype(np.float32), range(2, 7), seed=0)
    best_k = max(k_scores, key=lambda k: k_scores[k])
    assert best_k == 4

    labels = cluster_sites(features.astype(np.float32), n_clusters=best_k, seed=0)
    assert len(np.unique(labels)) == 4
