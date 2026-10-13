"""
hydrate_tweets.py

Fetches tweet text for a list of tweet IDs using the X (Twitter) API v2
Tweet Lookup endpoint (up to 100 IDs per request). Built for the UK MPs
hostility dataset (Zenodo 10.5281/zenodo.10809695), which ships only tweet
IDs and labels, not text, per Twitter's data-sharing terms.

Requires a BEARER_TOKEN environment variable (put it in a .env file — see
README). Run this on your own machine, NOT in a shared/committed script,
since it needs your paid API access.

Expected input CSV: at minimum a column of tweet IDs (default column name
"tweet_id" — adjust --id-col if the dataset uses a different name).

Usage:
    python scripts/hydrate_tweets.py \
        --input data/uk_mps_hostility_ids.csv \
        --id-col tweet_id \
        --output data/uk_mps_hostility_hydrated.csv
"""

import argparse
import csv
import os
import time

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # fine if python-dotenv isn't installed; env var can be set another way

API_URL = "https://api.twitter.com/2/tweets"
BATCH_SIZE = 100  # max IDs per request allowed by the endpoint
RATE_LIMIT_PAUSE_SECONDS = 2  # be conservative; adjust based on your tier's rate limit


def bearer_oauth(bearer_token):
    def auth(r):
        r.headers["Authorization"] = f"Bearer {bearer_token}"
        r.headers["User-Agent"] = "misinfo-project-hydration/1.0"
        return r
    return auth


def fetch_batch(ids_batch, bearer_token, tweet_fields="created_at,public_metrics,author_id"):
    params = {"ids": ",".join(str(i) for i in ids_batch), "tweet.fields": tweet_fields}
    resp = requests.get(API_URL, params=params, auth=bearer_oauth(bearer_token))
    if resp.status_code == 429:
        # Rate limited — wait for the reset window and retry once.
        reset = int(resp.headers.get("x-rate-limit-reset", time.time() + 60))
        wait = max(reset - time.time(), 5)
        print(f"  Rate limited. Waiting {wait:.0f}s...")
        time.sleep(wait)
        resp = requests.get(API_URL, params=params, auth=bearer_oauth(bearer_token))
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing tweet IDs")
    parser.add_argument("--id-col", default="tweet_id", help="Column name holding tweet IDs")
    parser.add_argument("--output", required=True, help="Where to save the hydrated CSV")
    args = parser.parse_args()

    bearer_token = os.environ.get("BEARER_TOKEN")
    if not bearer_token:
        raise SystemExit(
            "BEARER_TOKEN not set. Put it in a .env file (BEARER_TOKEN=...) "
            "or export it in your shell before running this script."
        )

    import pandas as pd

    df = pd.read_csv(args.input)
    ids = df[args.id_col].astype(str).tolist()
    print(f"Loaded {len(ids)} tweet IDs from {args.input}")

    hydrated = {}
    n_batches = (len(ids) + BATCH_SIZE - 1) // BATCH_SIZE

    # Write incrementally so a crash/rate-limit doesn't lose progress.
    file_exists = os.path.exists(args.output)
    with open(args.output, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["id", "text", "created_at", "author_id"])

        already_done = set()
        if file_exists:
            with open(args.output, "r", encoding="utf-8") as rf:
                reader = csv.reader(rf)
                next(reader, None)
                already_done = {row[0] for row in reader if row}

        for batch_num in range(n_batches):
            batch = ids[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
            batch = [i for i in batch if i not in already_done]
            if not batch:
                continue

            print(f"Batch {batch_num + 1}/{n_batches} ({len(batch)} IDs)...")
            try:
                data = fetch_batch(batch, bearer_token)
            except requests.HTTPError as e:
                print(f"  Error on batch {batch_num + 1}: {e}. Skipping batch.")
                continue

            for tweet in data.get("data", []):
                writer.writerow([
                    tweet["id"],
                    tweet.get("text", "").replace("\n", " "),
                    tweet.get("created_at", ""),
                    tweet.get("author_id", ""),
                ])
                hydrated[tweet["id"]] = True

            f.flush()
            time.sleep(RATE_LIMIT_PAUSE_SECONDS)

    n_missing = len(ids) - len(hydrated) - len(already_done)
    print(f"\nDone. Hydrated {len(hydrated)} new tweets this run.")
    print(f"Missing/unavailable (deleted, suspended, protected): ~{max(n_missing, 0)}")
    print(f"Saved to {args.output}")
    print(
        "\nNote: merge this back with the original label CSV on tweet ID "
        "before training — this script only fetches text, not labels."
    )


if __name__ == "__main__":
    main()
