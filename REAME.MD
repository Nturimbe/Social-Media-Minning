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
- [ ] BERT fine-tuned classifier
- [ ] Run UPFD graph analysis + GNN classifier (Colab, external to this repo's sandbox)
- [ ] Joint analysis merging NLP + network results
- [ ] Threat/defamation leg — data + modeling (not yet started)
- [ ] Full write-up (IMRaD format)

## Repository Structure

```
.
├── README.md
├── data/
│   └── processed_politifact.csv     # merged fake/real PolitiFact data + cascade_size
├── notebooks/
│   └── UPFD_network_analysis.ipynb  # run in Google Colab — pulls UPFD propagation graphs
├── scripts/
│   ├── build_dataset.py             # merges fake/real CSVs, computes cascade_size
│   ├── train_baseline.py            # TF-IDF + LogReg/SVM classifier
│   └── analyze_diffusion.py         # Mann-Whitney U test on cascade size
├── results/
│   └── baseline_results.json        # TF-IDF + LogReg/SVM accuracy & F1
└── docs/
    └── bibliography.md              # annotated bibliography — 6 papers + Vosoughi et al. (2018)
```

> Note: `FakeNewsNet/` (the cloned dataset repo) is not tracked in git — it's
> regenerated locally via `git clone https://github.com/KaiDMML/FakeNewsNet.git`.

## Data

**FakeNewsNet** (PolitiFact subset): 432 fake + 624 real political news
items, each with a title, source URL, and the IDs of tweets that shared the
story. Full retweet/user network data is not available without Twitter API
access (restricted since 2023), so:

- **NLP layer** uses the title text directly (available locally)
- **Network layer** uses two approaches:
  - A lightweight **cascade-size proxy** (tweet-count per story) computable
    locally
  - The pre-hydrated **UPFD** propagation-graph dataset (Dou et al., 2021),
    pulled via the Colab notebook in `notebooks/`, since it's hosted on
    Google Drive and out of this project's local sandbox network reach

## Methods So Far

- **Baseline classification:** TF-IDF (5,000 features, 1–2 grams) +
  Logistic Regression / Linear SVM on story titles. Linear SVM: **87.7%
  accuracy, F1 = 0.854**.
- **Diffusion proxy:** Mann-Whitney U test comparing cascade sizes (fake
  median = 79 tweets, real median = 8 tweets). **p < 0.001**, rank-biserial
  effect size ≈ 0.27 — statistically significant, moderate effect.

## Key Literature

Six core papers anchor the three legs of the taxonomy (full annotated
bibliography in `docs/bibliography.md`, in progress):

1. Sahid, Maleh & Ouazzane (2025) — bibliometric analysis of fake news research
2. Altuncu et al. (2025) — FACTors: fact-checking ecosystem dataset
3. Karami, Zain & Jamal (2025) — health misinformation SLR (definitional gaps)
4. Rossi (2024) — Paper II, bot/misinformation moderation-failure case study
5. Burke-Moore, Williams & Bright (2025) — large-scale online abuse of UK public figures
6. Rayhan, Boeriswati & Iskandar (2025) — authorship attribution on defamatory texts

Plus Vosoughi, Roy & Aral (2018, *Science*) as the foundational misinformation-diffusion study.

## Reproducing

```bash
git clone https://github.com/KaiDMML/FakeNewsNet.git
pip install pandas scikit-learn networkx matplotlib seaborn scipy --break-system-packages
python scripts/build_dataset.py       # merges fake/real, computes cascade_size
python scripts/train_baseline.py      # TF-IDF + LogReg/SVM
python scripts/analyze_diffusion.py   # Mann-Whitney U test on cascade size
```

For the network/graph layer, open `notebooks/UPFD_network_analysis.ipynb`
in Google Colab (GPU runtime recommended) and run top to bottom.

## Course Context

Social Media Mining, 300-level. Semester project designed to be scoped as a
testable, research-backed hypothesis rather than an open-ended "solve
misinformation" problem — extending known methods (Vosoughi et al. 2018;
UPFD/Dou et al. 2021) to a specific dataset and a novel three-way taxonomy
angle (misinformation vs. political threats vs. defamation).
