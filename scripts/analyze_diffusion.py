"""
analyze_diffusion.py

Compares cascade size (tweet-count proxy for diffusion reach) between fake
and real PolitiFact stories using a Mann-Whitney U test, since the
distribution is heavily right-skewed (a few mega-viral stories in both
classes) and not well suited to a t-test.

Usage:
    python scripts/analyze_diffusion.py
        [--data data/processed_politifact.csv]
        [--out results/diffusion_stats.json]
"""

import argparse
import json
import os

import pandas as pd
from scipy import stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed_politifact.csv")
    parser.add_argument("--out", default="results/diffusion_stats.json")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    fake_cascade = df[df["label"] == 1]["cascade_size"]
    real_cascade = df[df["label"] == 0]["cascade_size"]

    u_stat, p_value = stats.mannwhitneyu(fake_cascade, real_cascade, alternative="two-sided")

    n1, n2 = len(fake_cascade), len(real_cascade)
    effect_size = 1 - (2 * u_stat) / (n1 * n2)

    result = {
        "n_fake": n1,
        "n_real": n2,
        "median_cascade_fake": float(fake_cascade.median()),
        "median_cascade_real": float(real_cascade.median()),
        "mean_cascade_fake": float(fake_cascade.mean()),
        "mean_cascade_real": float(real_cascade.mean()),
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
        "rank_biserial_effect_size": float(effect_size),
    }

    print(f"Mann-Whitney U test: U={u_stat:.1f}, p={p_value:.6f}")
    print(f"Median cascade size - Fake: {result['median_cascade_fake']}, "
          f"Real: {result['median_cascade_real']}")
    print(f"Significant at alpha=0.05: {result['significant_at_0.05']}")
    print(f"Rank-biserial effect size: {effect_size:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
