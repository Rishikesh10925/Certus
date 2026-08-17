"""Extracts or derives site_id / camera_id per image (roadmap Section 3.6,
Section 4.1).

If the dataset ships real site/camera metadata, parse it directly. If not,
derive a proxy from EXIF tags, filename structure, or unsupervised
clustering of frozen-backbone features -- and record which strategy was
used, since the LOSO experiment's validity depends on sites being genuinely
distinct (verify with the domain-classifier diagnostic in Section 4.4:
AUC > 0.80 to proceed).

Week 1 audit (docs/logbook.md, docs/audit/cbri_audit_report_v1.json): CBRI
has no site_id/camera_id field, and a 50-image EXIF sample carried no GPS,
Make, or Model tags -- every tag came back empty. Filenames are bare numeric
IDs with a Roboflow-added content hash, no site encoding either. So none of
the cheap options apply here; the Week 2 plan is unsupervised clustering on
frozen-VLM embeddings, verified against the domain-classifier AUC gate
before it's trusted for LOSO.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.intp]


def choose_k(features: FloatArray, k_range: range, seed: int = 0) -> dict[int, float]:
    """Silhouette score per candidate cluster count, so the number of proxy
    sites is picked against a metric instead of guessed."""
    from sklearn.metrics import silhouette_score

    scores: dict[int, float] = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(features)
        scores[k] = float(silhouette_score(features, labels))
    return scores


def cluster_sites(features: FloatArray, n_clusters: int, seed: int = 0) -> IntArray:
    """KMeans over frozen-VLM features as a proxy site label. A cluster is
    not a site until `domain_classifier_auc` confirms it's separable."""
    labels: IntArray = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(
        features
    )
    return labels


def domain_classifier_auc(features: FloatArray, site_labels: IntArray, seed: int = 0) -> float:
    """Trains a logistic regression to predict site_labels from features,
    on a held-out split, and returns test-set AUC (macro one-vs-rest for
    3+ sites). This is the diagnostic gate: a site label -- real or
    clustered -- that isn't linearly separable in feature space isn't a
    genuinely distinguishable site for the LOSO study (roadmap Section
    4.4: AUC > 0.80 to proceed; below 0.65 is the Week 2 pivot trigger per
    the risk register).
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features, site_labels, test_size=0.3, random_state=seed, stratify=site_labels
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)
    classes = clf.classes_
    if len(classes) == 2:
        return float(roc_auc_score(y_test, probs[:, 1]))
    y_test_bin = label_binarize(y_test, classes=classes)
    return float(roc_auc_score(y_test_bin, probs, average="macro", multi_class="ovr"))
