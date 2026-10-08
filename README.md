# protein-hmm

An educational implementation of a Hidden Markov Model for protein sequence comparison, modelled after [HMMER's core architecture](https://github.com/EddyRivasLab/hmmer). It builds a profile from a multiple sequence alignment and scores protein sequences against that profile. A separate weather example introduces HMM inference and training.

## Set up a development environment

Install [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), then run from the repository root:

```bash
micromamba create -f environment.yml
micromamba activate protein-hmm
pre-commit install
```

The environment includes Python 3.14, NumPy, pytest, Ruff, and pre-commit. Run the tests and code checks with:

```bash
python -m pytest tests
pre-commit run --all-files
```

Without activating the environment, prefix commands with `micromamba run -n protein-hmm`.

## Project structure

- [protein_profile_hmm/](protein_profile_hmm/README.md): protein profile model, sequence parsers, dataset downloader, and scoring demo.
- [weather_example/](weather_example/README.md): introductory HMM, weather demo, and Baum-Welch training comparison.
- [data/](data/DATASETS.md): checked-in HMMER tutorial alignments and sequences.
- [tests/](tests/): unit and integration tests.
- [.github/workflows/](.github/workflows/): pull-request test automation.
