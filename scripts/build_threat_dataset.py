"""
build_threat_dataset.py

Downloads the Davidson et al. (2017) hate speech/offensive language dataset
and derives a political-context subset via keyword filtering, since no open
English dataset purpose-built for political threats is directly downloadable
in this project's sandbox (see docs/bibliography.md for the access-restricted
alternatives: the UK MPs hostility dataset and the Grimminger & Klinger 2020
US Election corpus).

Usage:
    python scripts/build_threat_dataset.py
        [--out-full data/davidson_hate_speech.csv]
        [--out-political data/political_threat_subset.csv]
"""

import argparse
import os
import re
import urllib.request

DAVIDSON_URL = (
    "https://raw.githubusercontent.com/t-davidson/"
    "hate-speech-and-offensive-language/master/data/labeled_data.csv"
)

POLITICAL_KEYWORDS = [
    "trump", "biden", "obama", "clinton", "president", "senator", "congress",
    "politician", "government", "gop", "democrat", "republican", "election",
    "vote", "senate", "mayor", "governor", "mp ", "parliament", "minister",
    "potus", "candidate",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-full", default="data/davidson_hate_speech.csv")
    parser.add_argument("--out-political", default="data/political_threat_subset.csv")
    args = parser.parse_args()

    import pandas as pd

    os.makedirs(os.path.dirname(args.out_full), exist_ok=True)
    urllib.request.urlretrieve(DAVIDSON_URL, args.out_full)

    df = pd.read_csv(args.out_full)
    print(f"Downloaded {len(df)} tweets to {args.out_full}")
    print(df["class"].value_counts().rename({0: "hate", 1: "offensive", 2: "neither"}))

    pattern = re.compile("|".join(POLITICAL_KEYWORDS), re.IGNORECASE)
    df["is_political"] = df["tweet"].str.contains(pattern, na=False)
    political_df = df[df["is_political"]].copy()

    political_df.to_csv(args.out_political, index=False)
    print(f"\nPolitical-context subset: {len(political_df)} tweets")
    print(political_df["class"].value_counts().rename({0: "hate", 1: "offensive", 2: "neither"}))
    print(f"Saved to {args.out_political}")


if __name__ == "__main__":
    main()
