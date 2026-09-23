"""
train_gk_classifier.py

Trains a hate/offensive speech (HOF) classifier on the Grimminger & Klinger
(2021) 2020 US Election corpus, and tests whether opposing a candidate
correlates with higher hate-speech rates than supporting them.

Dataset: Grimminger, L. & Klinger, R. (2021). Hate Towards the Political
Opponent. WASSA @ EACL 2021.
Free download: https://www.romanklinger.de/data-sets/GrimmingerKlingerWASSA2021.zip

Usage:
    python scripts/train_gk_classifier.py
        [--train data/gk_train.tsv]
        [--test data/gk_test.tsv]
        [--out-classifier results/gk_classifier_results.json]
        [--out-stance results/gk_stance_hate_analysis.json]
"""

import argparse
import json
import os

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC


def train_classifier(train, test, out_path):
    y_train = (train["HOF"] == "Hateful").astype(int)
    y_test = (test["HOF"] == "Hateful").astype(int)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train["text"])
    X_test = vectorizer.transform(test["text"])

    results = {}
    for name, model in [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("Linear SVM", LinearSVC(class_weight="balanced", max_iter=2000)),
    ]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        results[name] = {"accuracy": acc, "f1": f1}
        print(f"=== {name} ===")
        print(f"Accuracy: {acc:.3f}, F1 (Hateful): {f1:.3f}")
        print(classification_report(y_test, preds, target_names=["Non-Hateful", "Hateful"]))
        print()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved classifier results to {out_path}")


def stance_hate_analysis(df, out_path):
    df["is_hateful"] = df["HOF"] == "Hateful"
    results = {}

    for target in ["Trump", "Biden", "West"]:
        against_rate = df[df[target] == "Against"]["is_hateful"].mean()
        favor_rate = df[df[target] == "Favor"]["is_hateful"].mean()
        n_against = (df[target] == "Against").sum()
        n_favor = (df[target] == "Favor").sum()

        sub = df[df[target].isin(["Against", "Favor"])]
        contingency = pd.crosstab(sub[target], sub["is_hateful"])
        chi2, p, dof, expected = chi2_contingency(contingency)

        print(f"=== {target} ===")
        print(f"Hate rate when AGAINST {target}: {against_rate:.1%} (n={n_against})")
        print(f"Hate rate when FAVOR {target}:   {favor_rate:.1%} (n={n_favor})")
        print(f"Chi-square: chi2={chi2:.3f}, p={p:.6f}, significant={p < 0.05}")
        print()

        results[target] = {
            "against_hate_rate": float(against_rate), "n_against": int(n_against),
            "favor_hate_rate": float(favor_rate), "n_favor": int(n_favor),
            "chi2": float(chi2), "p_value": float(p), "significant": bool(p < 0.05),
        }

    overall_hate_rate = df["is_hateful"].mean()
    print(f"Overall corpus hate rate: {overall_hate_rate:.1%} (n={len(df)})")
    results["overall"] = {"hate_rate": float(overall_hate_rate), "n": len(df)}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved stance-hate analysis to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/gk_train.tsv")
    parser.add_argument("--test", default="data/gk_test.tsv")
    parser.add_argument("--out-classifier", default="results/gk_classifier_results.json")
    parser.add_argument("--out-stance", default="results/gk_stance_hate_analysis.json")
    args = parser.parse_args()

    train = pd.read_csv(args.train, sep="\t")
    test = pd.read_csv(args.test, sep="\t")
    train["text"] = train["text"].str.replace("[NEWLINE]", " ", regex=False)
    test["text"] = test["text"].str.replace("[NEWLINE]", " ", regex=False)

    train_classifier(train, test, args.out_classifier)

    full = pd.concat([train, test], ignore_index=True)
    stance_hate_analysis(full, args.out_stance)


if __name__ == "__main__":
    main()
