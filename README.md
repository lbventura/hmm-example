# protein-hmm

## Set up a development environment

Install [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), then create and activate the environment defined in `environment.yml`:

```bash
micromamba create -f environment.yml
micromamba activate protein-hmm
```

The environment includes Python 3.14, NumPy, pytest, Ruff, and pre-commit. To run commands without activating it, use `micromamba run -n protein-hmm <command>`.

## Run checks

```bash
micromamba run -n protein-hmm pytest tests
micromamba run -n protein-hmm ruff check .
```

## Introductory example (weather_example.py)

A small Hidden Markov Model (HMM) demo using the classic Eisner (2002) weather
example: hidden states `Sunny`/`Rainy`, observations `Walk`/`Shop`/`Clean`.

- `hmm.py` — `HiddenMarkovModel` with `forward`, `viterbi`, `baum_welch`, `sample`.
- `weather_example.py` — runs forward, Viterbi, and a naive single-sequence Baum-Welch.
- `compare_training.py` — naive vs. an *improved* Baum-Welch trainer, scored by
  the L1 distance between learned and true `(A, B)`.
- `tests/test_hmm.py` — pytest suite for the model.

```bash
python weather_example.py              # forward / Viterbi / single-sequence training
python compare_training.py  # naive vs improved Baum-Welch (≈4–5× lower L1)
python -m pytest tests/     # unit tests
```

## Improvements to Baum-Welch

Naive Baum-Welch (`HiddenMarkovModel.baum_welch`) trains on one short sequence
from one random init with no smoothing, in raw probability space.
`compare_training.py:baum_welch_improved` layers in four standard fixes:

- **Multi-sequence training.** Likelihood and sufficient statistics
  (`γ`, `ξ`) are accumulated across many i.i.d. sequences sampled from the same
  generative model. A single short sequence undersamples both transitions and
  emissions, so the M-step over-fits idiosyncrasies; pooling many sequences
  drives the estimator toward the population MLE.
- **Multiple random restarts.** The EM objective is non-convex with many local
  optima (and label-switching symmetries). We run EM from several independent
  random initializations and keep the model with the highest *total*
  log-likelihood across all training sequences, which sharply reduces the
  probability of returning a bad local maximum.
- **Additive Dirichlet (Laplace) smoothing in the M-step.** Each parameter
  update is `(count + α) / (sum_of_counts + α · K)` with a per-cell pseudo-count
  `α = pseudo` (here `0.5`). This is the MAP update under a symmetric
  `Dirichlet(α + 1)` prior and guarantees every entry of `π`, `A`, `B` stays
  *strictly* positive — so EM cannot collapse a row to a 0/1 vector it can
  never escape from, and the resulting `log(0)`s never appear in the next E-step.
- **Log-space forward/backward with `logsumexp`.** `α_t` and `β_t` shrink
  geometrically with sequence length, so on `T ≈ 100+` they underflow to
  zero in plain probability space. Working in log-space and combining terms with
  `logsumexp(x) = max(x) + log(sum(exp(x - max(x))))` keeps everything finite
  and gives exact log-likelihoods regardless of sequence length.

## Why Dirichlet initialization?

Both the naive and improved trainers initialise `π`, `A`, `B` by sampling each
row from a `Dirichlet(1, …, 1)`. This is the small but important fifth
ingredient:

- **Always a valid distribution.** A Dirichlet draw is by construction
  non-negative and sums to 1, so every initial `π`, every row of `A`, and every
  row of `B` is already a proper categorical distribution — no projection,
  clipping, or renormalisation step is needed.
- **Uniform over the simplex (concentration 1).** `Dirichlet(1, …, 1)` is the
  flat distribution on the probability simplex, so the initializer has no
  built-in preference for any hidden state and no hidden bias toward a
  particular transition or emission pattern.
- **Asymmetric, diverse starts → useful restarts.** Unlike `1/N` uniform
  initialisation (which is symmetric under any permutation of hidden states,
  leaving EM unable to break the tie), each Dirichlet draw is generically
  asymmetric. Different restarts therefore land in different basins of
  attraction, which is the whole point of doing multiple restarts.
- **Strictly positive entries.** With probability 1 the draw has no zeros, so
  `log π`, `log A`, `log B` are finite from the first iteration and the
  log-space E-step is well-defined immediately, with no special-casing of
  initial zeros.

## Protein Profile HMM (`protein_profile_hmm.py`)

A glocal profile Hidden Markov Model for protein sequence comparison,
modelled after HMMER's core architecture.  A profile is built from a
multiple sequence alignment (MSA) and then used to score query sequences
against the represented protein family.

Architecture — the profile has K Match positions (one per consensus
column), K Delete states (silent), and K+1 Insert states.  All emission
and transition probabilities are stored in **log2** space; scoring returns
log-odds bit scores relative to a background null model—positive scores
mean the query fits the family better than a random sequence.

### State transitions

There are 9 slots per column slot k:

| Index | Transition |
|-------|-----------|
| 0 | M_k → M_{k+1} |
| 1 | M_k → I_k |
| 2 | M_k → D_{k+1} |
| 3 | I_k → M_{k+1} |
| 4 | I_k → I_k |
| 5 | D_k → M_{k+1} |
| 6 | D_k → D_{k+1} |
| 7 | B → M_1 (begin) |
| 8 | B → D_1 (begin) |

### Key methods

- `ProfileHMM.build_from_msa(msa)` counts emissions and
transitions, applies Laplace smoothing, and returns a trained profile.
- `viterbi(seq)` returns the maximum-likelihood path bit score;
- `forward(seq)` marginalises over all paths via the Forward algorithm.

### Imports

- `amino_acids` — 20-letter canonical alphabet, residue encoding, and
  Robinson & Robinson (1991) background frequencies used to convert
  log-probabilities into log-odds scores.
- `msa` — Stockholm-format MSA parser and the `match_column_mask` function
  that assigns alignment columns to Match or Insert slots via HMMER's
  `--symfrac` rule.
