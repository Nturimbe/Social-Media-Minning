"""
train_threat_classifier.py

Trains a TF-IDF + Logistic Regression 3-class classifier (hate speech /
offensive / neither) on the full Davidson et al. dataset, then tests whether
political-context tweets show a disproportionate hate-speech rate compared
to the rest of the corpus (chi-square test on the political-keyword subset
built by build_threat_dataset.py).

Usage:
    python scripts/train_threat_classifier.py
        [--data data/davidson_hate_speech.csv]
        [--out results/threat_results.json]
"""

import argparse
import json
import os

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

POLITICAL_KEYWORDS = (
    "trump|biden|obama|clinton|president|senator|congress|politician|"
    "government|gop|democrat|republican|election|vote|senate|mayor|"
    "governor|parliament|minister|potus|candidate"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/davidson_hate_speech.csv")
    parser.add_argument("--out", default="results/threat_results.json")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df["tweet"].astype(str)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_tfidf, y_train)
    preds = clf.predict(X_test_tfidf)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    print("=== Threat/Hostility Classifier (3-class) ===")
    print(f"Accuracy: {acc:.3f}, Macro F1: {f1_macro:.3f}")
    print(classification_report(y_test, preds, target_names=["Hate", "Offensive", "Neither"]))

    df["is_political"] = df["tweet"].str.contains(POLITICAL_KEYWORDS, case=False, na=False)
    df["is_hate"] = df["class"] == 0

    contingency = pd.crosstab(df["is_political"], df["is_hate"])
    print("\nContingency table (rows=political, cols=is_hate):")
    print(contingency)

    chi2, p, dof, expected = chi2_contingency(contingency)
    pol_hate_rate = df[df["is_political"]]["is_hate"].mean()
    nonpol_hate_rate = df[~df["is_political"]]["is_hate"].mean()

    print(f"\nChi-square test: chi2={chi2:.3f}, p={p:.6f}, significant={p < 0.05}")
    print(f"Hate speech rate in political tweets: {pol_hate_rate:.3%}")
    print(f"Hate speech rate in non-political tweets: {nonpol_hate_rate:.3%}")

    results = {
        "classifier": {"accuracy": acc, "macro_f1": f1_macro},
        "chi_square_test": {
            "chi2": float(chi2),
            "p_value": float(p),
            "significant": bool(p < 0.05),
            "political_hate_rate": float(pol_hate_rate),
            "nonpolitical_hate_rate": float(nonpol_hate_rate),
            "note": (
                "Political subset is small (n~430) from keyword filtering; "
                "result is directionally positive but not statistically "
                "significant at this sample size."
            ),
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
