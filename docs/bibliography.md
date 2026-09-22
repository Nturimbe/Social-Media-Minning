# Annotated Bibliography

Core literature anchoring the project's three-way taxonomy: **misinformation**,
**political threats**, and **defamation** on social media. Entries are grouped
by which leg of the taxonomy they primarily support.

---

## Foundational

### Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. *Science*, 359(6380), 1146–1151.

The landmark MIT study analyzing ~126,000 rumor cascades on Twitter, showing
that false political news spreads significantly faster, farther, and deeper
than true news. Establishes the diffusion-based framing this project's
cascade-size analysis directly extends. Foundational citation for the
misinformation-diffusion research question.

---

## Misinformation

### 1. Sahid, A., Maleh, Y., & Ouazzane, K. (2025). Changing landscape of fake news research on social media: a bibliometric analysis. *Quality & Quantity*, 59(Suppl 2), 901–954.

A large-scale bibliometric study (co-citation and bibliographic linkage
analysis across Scopus and Web of Science) mapping the evolution of fake-news
research on social media. Flags a persistent methodological gap: detection
methods still struggle with **limited labeled data**, and surveys emerging
concealment tactics used to evade detection systems. Used as the project's
"state of the field" citation for the misinformation leg.

### 2. Altuncu, E., Başkent, C., Bhattacherjee, S., Li, S., & Roy, D. (2025). FACTors: A New Dataset for Studying the Fact-checking Ecosystem. *Proceedings of the 48th International ACM SIGIR Conference*, 3530–3539.

Introduces a dataset spanning fact-checking reports from many organizations
over a long time period, and uses it to examine the **political leanings of
fact-checking organizations** and to build a credibility-scoring method.
Relevant because it treats fact-checking itself as a system with potential
biases worth interrogating, rather than as a neutral ground-truth oracle —
directly useful for the project's argument that platforms and moderators
struggle with ambiguity, not just detection accuracy.

### 3. Karami, A., Zain, A., & Jamal, A. (2025). Unveiling the information mirage: a systematic literature review of health misinformation on social media. *Journal of Public Health*.

A systematic review of 192 studies (2010–2022) on health misinformation.
Central finding: **53% of studies never gave a clear definition** of
"misinformation," and among those that did, terms like misinformation,
disinformation, and infodemic were used inconsistently across the field.
Direct evidence that even within one well-studied category, researchers
struggle to define terms precisely — supporting this project's premise that
a three-way taxonomy (misinformation/threat/defamation) is a real,
underserved distinction. Also confirms the field splits into two dominant
research themes — **diffusion** and **detection/mitigation** — which this
project's own structure mirrors.

### 4. Rossi, S. (2024). Bots on Social Media: The Past, Present and Future [Paper II: The Scamdemic Conspiracy Theory and Twitter's Failure to Moderate COVID-19 Misinformation]. PhD dissertation, Copenhagen Business School.

A case study of 8,263 tweets using the keyword "scamdemic" during March 2021,
combining network analysis (betweenness/in/out-degree centrality) with the
Botometer bot-detection tool to identify influential misinformation-spreading
accounts. Finding: only ~21% of influential accounts were bots — most spread
was organic and human-driven — and only 12.7% of clearly policy-violating
accounts were suspended after two months despite Twitter's stated moderation
policy. Useful as a real case study of platform moderation failure and
ambiguity, and candidly documents the post-2023 Twitter/X API paywall that
motivates this project's use of cascade-size proxies and the UPFD dataset
instead of live network hydration.

---

## Political Threats

### 5. Burke-Moore, L., Williams, A. R., & Bright, J. (2025). Journalists are most likely to receive abuse: analysing online abuse of UK public figures across sport, politics, and journalism on Twitter. *EPJ Data Science*, 14, 32.

A large-scale study from the Alan Turing Institute analyzing 45.5 million
tweets directed at 4,602 UK public figures across three domains (MPs,
footballers, journalists), labeled with fine-tuned transformer models. Finds
MPs receive more abuse in raw volume, but journalists are most likely to
receive abuse once other factors are controlled for; abuse is highly
concentrated among a small number of individuals; and a more prominent
online presence plus being male predicts higher abuse levels across all
groups. Provides a rigorous, comparable, multi-domain framework for the
political-threat leg of the taxonomy.

---

## Defamation

### 6. Rayhan, M., Boeriswati, E., & Iskandar, I. (2025). Authorship Attribution on Anonymous Defamatory Texts in Social Media: A Systematic Review. Universitas Negeri Jakarta.

A systematic review of 50 Scopus-indexed journal articles (2020–2025) on
identifying authorship of anonymous defamatory content on social media.
Finds the field has shifted decisively toward transformer-based models
(BERT, mBERT, XLM-RoBERTa), which perform well on multilingual and
short-text cases, but flags a real transparency problem: these models'
outputs are hard to explain, which matters far more for a legal category
like defamation than for less consequential classification tasks. Anchors
the defamation leg of the taxonomy and motivates a discussion of
explainability as a shared weak point across all three categories.

---

## How These Fit Together

| Paper | Taxonomy leg | Contribution |
|---|---|---|
| Vosoughi et al. (2018) | Misinformation | Foundational diffusion finding |
| Sahid, Maleh & Ouazzane (2025) | Misinformation | Field-level state of the art |
| Altuncu et al. (2025) | Misinformation | Fact-checking system bias |
| Karami, Zain & Jamal (2025) | Misinformation | Definitional inconsistency |
| Rossi (2024), Paper II | Misinformation | Moderation-failure case study |
| Burke-Moore, Williams & Bright (2025) | Political threats | Large-scale abuse detection |
| Rayhan, Boeriswati & Iskandar (2025) | Defamation | Authorship attribution, explainability |

The recurring thread across all six: **detection systems are maturing
technically (transformers, large-scale datasets) faster than the field is
maturing conceptually (clear definitions, cross-category distinctions,
explainability)**. This is the gap the project's taxonomy angle targets.
