"""
build_dataset.py

Merges the PolitiFact fake/real CSVs from the FakeNewsNet repo, labels them,
and computes a cascade-size diffusion proxy (number of tweets sharing each
story) since full retweet/user network data requires Twitter API access we
don't have.

Prerequisite:
    git clone https://github.com/KaiDMML/FakeNewsNet.git

Usage:
    python scripts/build_dataset.py
        [--fnn-dir FakeNewsNet/dataset]
        [--out data/processed_politifact.csv]
"""

import argparse
import os

import pandas as pd


def count_tweets(tweet_ids_field: str) -> int:
    """Count tab-separated tweet IDs in a FakeNewsNet tweet_ids field."""
    if pd.isna(tweet_ids_field):
        return 0
    return len(str(tweet_ids_field).split("\t"))


def build_dataset(fnn_dir: str) -> pd.DataFrame:
    fake_path = os.path.join(fnn_dir, "politifact_fake.csv")
    real_path = os.path.join(fnn_dir, "politifact_real.csv")

    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)

    fake["label"] = 1  # fake
    real["label"] = 0  # real

    df = pd.concat([fake, real], ignore_index=True)
    df = df.dropna(subset=["title"])

    df["cascade_size"] = df["tweet_ids"].apply(count_tweets)

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fnn-dir",
        default="FakeNewsNet/dataset",
        help="Path to the cloned FakeNewsNet dataset directory",
    )
    parser.add_argument(
        "--out",
        default="data/processed_politifact.csv",
        help="Output path for the merged, processed CSV",
    )
    args = parser.parse_args()

    df = build_dataset(args.fnn_dir)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Merged dataset: {len(df)} rows")
    print(df.groupby("label").size().rename({0: "real", 1: "fake"}))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
