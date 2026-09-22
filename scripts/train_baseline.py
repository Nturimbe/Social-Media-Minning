"""
train_baseline.py

Trains baseline TF-IDF + Logistic Regression / Linear SVM classifiers on
PolitiFact story titles (fake vs. real) and saves accuracy/F1 to JSON.

Usage:
    python scripts/train_baseline.py
        [--data data/processed_politifact.csv]
        [--out results/baseline_results.json]
"""

import argparse
import json
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed_politifact.csv")
    parser.add_argument("--out", default="results/baseline_results.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df["title"].astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced", max_iter=2000),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        results[name] = {"accuracy": acc, "f1": f1}

        print(f"=== {name} ===")
        print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}")
        print(classification_report(y_test, preds, target_names=["Real", "Fake"]))
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
