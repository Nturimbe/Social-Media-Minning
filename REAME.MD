# Distinguishing Misinformation, Political Threats & Defamation on Social Media

Semester project for a 300-level Social Media Mining course. The goal is a
research-backed problem, drilled down into an actual pipeline: literature
review → data → models → statistical evaluation → write-up, at a standard
that could plausibly be submitted as a short academic paper.

## Problem Statement

Social media platforms, moderators, and researchers routinely lump together
three distinct categories of harmful content:

- **Misinformation** — false or misleading claims, not necessarily targeting
  a person
- **Political threats** — language signaling intent to harm a public
  official or figure
- **Defamation** — a false *statement of fact* (not opinion) that harms a
  specific person's reputation, a legally defined category

Most detection systems handle exactly one of these in isolation. Very few
attempt to distinguish between them, even though the distinction matters a
great deal in practice (a harsh true claim isn't defamation; a false claim
about policy isn't necessarily a threat; an angry insult isn't necessarily
either). This project investigates that gap.

## Research Question & Hypothesis

**RQ:** Does politically false news spread differently — faster, wider, and
within more structurally clustered communities — than true news on social
media, and how does this compare to the distinct but often-conflated
categories of political threats and defamation?

**H1:** False political news exhibits greater cascade reach and velocity
than true political news.
**H0:** No significant difference in diffusion patterns between true and
false political news.

(This mirrors Vosoughi, Roy & Aral, 2018, *Science*, extended with a network
layer and situated against the threat/defamation taxonomy above.)

## Status

Work in progress. Current milestones:

- [x] Topic scoped and research question locked
- [x] Literature review — 6 core papers identified and verified (see below)
- [x] Dataset acquired (FakeNewsNet — PolitiFact subset)
- [x] Baseline NLP classifier (title-only, TF-IDF + Logistic Regression / SVM)
- [x] Cascade-size diffusion proxy computed and statistically tested
- [x] Colab notebook prepared for full network/graph analysis (UPFD)
- [x] Reproducible scripts (`build_dataset.py`, `train_baseline.py`, `analyze_diffusion.py`)
- [x] Annotated bibliography (`docs/bibliography.md`)
- [x] Threat-leg dataset acquired — Davidson et al. proxy (supplementary)
      AND Grimminger & Klinger (2021) real political corpus (primary)
- [x] Threat/hostility classifier trained on both; GK is primary
      (88.8% accuracy, F1 Hateful = 0.396)
- [x] Statistically significant threat-leg finding: opposing a candidate
      correlates with higher hate-speech rates (p < 0.01 for both Trump and Biden)
- [x] Defamation-leg dataset search exhausted for English — documented as a field-level
      gap; Japanese dataset (Kubota, 2023) used as a cross-lingual literature note only
- [ ] BERT fine-tuned classifier (misinformation leg)
- [ ] Run UPFD graph analysis + GNN classifier (Colab, external to this repo's sandbox)
- [ ] Joint analysis merging NLP + network results (misinformation leg)
- [ ] Expand political-context filtering for the threat leg (current keyword
      subset is small, n≈430 — chi-square result is directional, not significant)
- [ ] Full write-up (IMRaD format)

## Repository Structure

```
.
├── README.md
├── data/
│   ├── processed_politifact.csv       # misinformation leg: merged fake/real + cascade_size
│   ├── gk_train.tsv                   # threat leg (primary): GK 2020 election train split
│   ├── gk_test.tsv                    # threat leg (primary): GK 2020 election test split
│   ├── gk_readme.txt                  # GK dataset column/label documentation
│   ├── davidson_hate_speech.csv       # threat leg (supplementary): full hate-speech corpus
│   └── political_threat_subset.csv    # threat leg (supplementary): political-keyword subset
├── notebooks/
│   └── UPFD_network_analysis.ipynb    # run in Google Colab — pulls UPFD propagation graphs
├── scripts/
│   ├── build_dataset.py               # misinformation: merges fake/real CSVs, computes cascade_size
│   ├── train_baseline.py              # misinformation: TF-IDF + LogReg/SVM classifier
│   ├── analyze_diffusion.py           # misinformation: Mann-Whitney U test on cascade size
│   ├── train_gk_classifier.py         # threat (primary): HOF classifier + stance-hate chi-square
│   ├── build_threat_dataset.py        # threat (supplementary): downloads Davidson corpus, builds political subset
│   ├── train_threat_classifier.py     # threat (supplementary): 3-class classifier + chi-square test
│   └── hydrate_tweets.py              # threat (extended, optional): hydrates UK MPs dataset via X API v2
├── results/
│   ├── baseline_results.json          # misinformation: TF-IDF + LogReg/SVM accuracy & F1
│   ├── gk_classifier_results.json     # threat (primary): HOF classifier accuracy & F1
│   ├── gk_stance_hate_analysis.json   # threat (primary): opposition-vs-hostility chi-square results
│   └── threat_results.json            # threat (supplementary): Davidson classifier + chi-square
└── docs/
    ├── bibliography.md                # annotated bibliography — 8 papers/datasets + Vosoughi et al. (2018)
    └── GK_AnnotationGuidelines.pdf    # original GK annotation guidelines (for methods section)
```

> Note: `FakeNewsNet/` (the cloned dataset repo) is not tracked in git — it's
> regenerated locally via `git clone https://github.com/KaiDMML/FakeNewsNet.git`.

## Data

**Misinformation leg — FakeNewsNet (PolitiFact subset):** 432 fake + 624 real
political news items, each with a title, source URL, and the IDs of tweets
that shared the story. Full retweet/user network data is not available
without Twitter API access (restricted since 2023), so:

- **NLP layer** uses the title text directly (available locally)
- **Network layer** uses two approaches: a lightweight **cascade-size
  proxy** (tweet-count per story, computable locally) and the pre-hydrated
  **UPFD** propagation-graph dataset (Dou et al., 2021), pulled via the
  Colab notebook in `notebooks/` since it's hosted on Google Drive and out
  of this project's local sandbox network reach

**Threat leg — Grimminger & Klinger (2021), 2020 US Election corpus:**
3,000 tweets (2,400 train / 600 test, provided split), full text, annotated
for stance toward Trump/Biden/West and hate/offensive speech (HOF). Free,
direct download, no API required — this is the project's primary threat-leg
dataset, replacing the earlier Davidson et al. keyword-filtered proxy
(retained in the repo as a supplementary comparison). The UK MPs hostility
dataset was also investigated but requires paid X/Twitter API access to
hydrate (see `scripts/hydrate_tweets.py` if you want to pursue it later —
X API moved to pay-per-use pricing in Feb 2026, no free reads at all).

**Defamation leg — no open English dataset found.** A systematic check
against a comprehensive 2025 survey of English hate-speech/abuse datasets
confirmed none target defamation specifically — a real, citable gap in the
field, not just this project. A Japanese Twitter dataset (Kubota, 2023) is
used as a cross-lingual literature note only (see `docs/bibliography.md`);
no local modeling is done on this leg.

## Methods So Far

**Misinformation:**
- **Baseline classification:** TF-IDF (5,000 features, 1–2 grams) +
  Logistic Regression / Linear SVM on story titles. Linear SVM: **87.7%
  accuracy, F1 = 0.854**.
- **Diffusion proxy:** Mann-Whitney U test comparing cascade sizes (fake
  median = 79 tweets, real median = 8 tweets). **p < 0.001**, rank-biserial
  effect size ≈ 0.27 — statistically significant, moderate effect.

**Threats:**
- **Primary dataset (Grimminger & Klinger, 2021):** TF-IDF + Logistic
  Regression / Linear SVM, hateful vs. non-hateful (HOF). Linear SVM:
  **88.8% accuracy, F1 (Hateful) = 0.396** — accuracy is high but hate-class
  F1 is modest, an honest reflection of severe class imbalance (only ~12%
  of tweets are labeled Hateful) rather than a modeling failure.
- **Stance-hostility finding:** tweets **against** a candidate show
  significantly higher hate rates than tweets **favoring** them — Trump:
  18.5% vs. 12.9% (p = 0.0026); Biden: 26.7% vs. 11.4% (p < 0.0001). West's
  subset was too small (n=6) to test. This is the project's strongest
  result so far — statistically significant, not just directional.
- **Supplementary comparison (Davidson et al. proxy):** 3-class classifier
  (hate/offensive/neither) on a general hate-speech corpus, 85.4% accuracy,
  macro F1 = 0.722. Political-keyword subset showed a directionally higher
  hate rate (7.83% vs. 5.74%) but was not significant (p = 0.133,
  underpowered at n≈430) — kept as a documented limitation/comparison point,
  now superseded by the Grimminger & Klinger result above.

## Key Literature

Eight core papers/datasets anchor the three legs of the taxonomy (full
annotated bibliography in `docs/bibliography.md`):

1. Sahid, Maleh & Ouazzane (2025) — bibliometric analysis of fake news research
2. Altuncu et al. (2025) — FACTors: fact-checking ecosystem dataset
3. Karami, Zain & Jamal (2025) — health misinformation SLR (definitional gaps)
4. Rossi (2024) — Paper II, bot/misinformation moderation-failure case study
5. Burke-Moore, Williams & Bright (2025) — large-scale online abuse of UK public figures
6. Grimminger & Klinger (2021) — 2020 US Election hate/stance corpus (used directly in modeling)
7. Rayhan, Boeriswati & Iskandar (2025) — authorship attribution on defamatory texts
8. Kubota (2023) — Japanese defamation taxonomy (cross-lingual literature note)

Plus Vosoughi, Roy & Aral (2018, *Science*) as the foundational misinformation-diffusion study.

## Setup: UK MPs Hostility Dataset (Threat Leg, Extended)

This dataset (Gorrell et al., Zenodo 10.5281/zenodo.10809695) ships only
tweet IDs and hostility labels, not text — hydration requires your own paid
X/Twitter API access (the free tier no longer supports reads as of 2026).

1. Request Zenodo access with your academic email.
2. Create an X developer account and subscribe to at least the Basic tier
   (~$200/mo) or pay-as-you-go, if available. Generate a Bearer Token.
3. `pip install pandas requests python-dotenv --break-system-packages`
4. Create a `.env` file with `BEARER_TOKEN=your_token_here` and add `.env`
   to `.gitignore` — never commit API keys.
5. Once your Zenodo request is approved, place the dataset in `data/` and run:
   ```bash
   python scripts/hydrate_tweets.py \
       --input data/uk_mps_hostility_ids.csv \
       --id-col tweet_id \
       --output data/uk_mps_hostility_hydrated.csv
   ```
6. Expect some tweets to come back missing (deleted/suspended/protected
   accounts) — this is a known, citable limitation (see Rossi, 2024 in
   `docs/bibliography.md`, which found only 12.7% of rule-violating accounts
   were suspended within two months, so most missing tweets are more likely
   user-deleted than platform-enforced).
7. Merge the hydrated text back with the original label file on tweet ID,
   then reuse the pattern in `scripts/train_threat_classifier.py`.

## Reproducing

```bash
git clone https://github.com/KaiDMML/FakeNewsNet.git
pip install pandas scikit-learn networkx matplotlib seaborn scipy --break-system-packages

# Misinformation leg
python scripts/build_dataset.py       # merges fake/real, computes cascade_size
python scripts/train_baseline.py      # TF-IDF + LogReg/SVM
python scripts/analyze_diffusion.py   # Mann-Whitney U test on cascade size

# Threat leg (primary — download the free GK zip yourself first:
# https://www.romanklinger.de/data-sets/GrimmingerKlingerWASSA2021.zip)
python scripts/train_gk_classifier.py     # HOF classifier + stance-hate chi-square

# Threat leg (supplementary comparison)
python scripts/build_threat_dataset.py    # downloads Davidson corpus, builds political subset
python scripts/train_threat_classifier.py # classifier + political-vs-rest chi-square test
```

For the network/graph layer, open `notebooks/UPFD_network_analysis.ipynb`
in Google Colab (GPU runtime recommended) and run top to bottom.

## Course Context

Social Media Mining, 300-level. Semester project designed to be scoped as a
testable, research-backed hypothesis rather than an open-ended "solve
misinformation" problem — extending known methods (Vosoughi et al. 2018;
UPFD/Dou et al. 2021) to a specific dataset and a novel three-way taxonomy
angle (misinformation vs. political threats vs. defamation).
