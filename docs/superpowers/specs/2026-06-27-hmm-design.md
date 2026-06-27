# Hidden Markov Model — Design Spec

**Date:** 2026-06-27  
**Author:** Copilot (autonomous)

## Overview

A simple, educational Hidden Markov Model implemented from scratch in Python with NumPy.
Uses the classic "weather" domain: hidden weather states drive observable human activities.

## Domain

- **Hidden states:** Sunny, Rainy
- **Observations:** Walk, Shop, Clean
- **Goal:** Demonstrate the three core HMM algorithms on a concrete, relatable example

## Architecture

### `hmm.py` — `HiddenMarkovModel` class

| Component | Description |
|-----------|-------------|
| `pi` | Initial state probability vector (shape: N) |
| `A` | Transition matrix (shape: N×N) |
| `B` | Emission matrix (shape: N×M) |

### Algorithms

1. **Forward algorithm** — compute P(observations | model) via dynamic programming
2. **Viterbi algorithm** — find the most likely hidden state sequence (argmax decoding)
3. **Baum-Welch algorithm** — unsupervised learning of A, B, pi from observations (EM)

### `main.py` — Demo script

Runs the weather example end-to-end:
1. Define the known model (pi, A, B)
2. Generate a synthetic observation sequence
3. Run forward algorithm → print likelihood
4. Run Viterbi → print decoded state sequence
5. Run Baum-Welch from random init → show parameters converge toward truth

## Dependencies

- Python 3.14
- NumPy

## Environment

- micromamba environment: `hmm-example`
- `environment.yml` defines the env for reproducibility

## Success Criteria

- All three algorithms produce correct results on the weather example
- Code is readable with docstrings explaining each algorithm
- Running `python main.py` produces clear, labelled output
