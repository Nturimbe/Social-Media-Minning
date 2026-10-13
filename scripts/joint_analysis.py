"""
joint_analysis.py

Merges the NLP and network layers for the misinformation leg: tests whether
the classifier's confidence that a story is fake correlates with how far it
spread (cascade_size). This is the "does classifier confidence correlate
with cascade speed" step from the project plan — the piece that makes the
result a joint finding rather than two separate, unconnected analyses.

NOTE: this reproduces the initial all-stories analysis. Many stories (especially real ones) have zero
recorded tweets; notebooks/01_full_analysis.ipynb adds the zero-inflation robustness checks that qualify
this result, and the paper reports those conclusions.

Usage:
    python scripts/joint_analysis.py
        [--data data/processed_politifact.csv]
        [--out results/joint_analysis_results.json]
"""

import argparse
import json
import os

import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed_politifact.csv")
    parser.add_argument("--out", default="results/joint_analysis_results.json")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df["title"].astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Calibrated SVM to get probability estimates (predicted "fakeness" confidence)
    clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=2000), cv=5)
    clf.fit(X_train_tfidf, y_train)
    fake_proba = clf.predict_proba(X_test_tfidf)[:, 1]  # P(fake)

    test_df = df.loc[idx_test].copy()
    test_df["fake_proba"] = fake_proba

    rho, p = spearmanr(test_df["fake_proba"], test_df["cascade_size"])
    print(f"Spearman correlation (P(fake) vs cascade_size): rho={rho:.3f}, p={p:.4f}")

    fake_only = test_df[test_df["label"] == 1]
    rho_fake, p_fake = spearmanr(fake_only["fake_proba"], fake_only["cascade_size"])
    print(f"Within fake stories only: rho={rho_fake:.3f}, p={p_fake:.4f}, n={len(fake_only)}")

    results = {
        "joint_analysis": {
            "description": (
                "Spearman correlation between classifier P(fake) confidence "
                "and cascade_size (diffusion proxy)"
            ),
            "all_test_stories": {"rho": float(rho), "p_value": float(p), "n": len(test_df)},
            "fake_stories_only": {
                "rho": float(rho_fake), "p_value": float(p_fake), "n": len(fake_only)
            },
        }
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
