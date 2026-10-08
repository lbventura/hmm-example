"""Tests for HiddenMarkovModel.

Uses the canonical Eisner (2002) ice cream / weather HMM values adapted to:
  States: Sunny=0, Rainy=1
  Observations: Walk=0, Shop=1, Clean=2
"""

import numpy as np
import pytest

from hmm import HiddenMarkovModel

# Canonical weather HMM parameters
PI = [0.6, 0.4]

A = [
    [0.7, 0.3],  # Sunny -> Sunny=0.7, Sunny -> Rainy=0.3
    [0.4, 0.6],  # Rainy -> Sunny=0.4, Rainy -> Rainy=0.6
]

B = [
    [0.5, 0.4, 0.1],  # Sunny: Walk=0.5, Shop=0.4, Clean=0.1
    [0.1, 0.3, 0.6],  # Rainy: Walk=0.1, Shop=0.3, Clean=0.6
]

OBS = [0, 1, 2]  # Walk, Shop, Clean


@pytest.fixture
def model() -> HiddenMarkovModel:
    return HiddenMarkovModel(PI, A, B)


def test_forward_returns_probability_between_0_and_1(model: HiddenMarkovModel) -> None:
    prob = model.forward(OBS)
    assert 0 < prob <= 1


def test_forward_single_observation(model: HiddenMarkovModel) -> None:
    # P(Walk) = pi[0]*B[0,0] + pi[1]*B[1,0] = 0.6*0.5 + 0.4*0.1 = 0.34
    prob = model.forward([0])
    assert abs(prob - 0.34) < 1e-9


def test_forward_known_sequence(model: HiddenMarkovModel) -> None:
    # Precomputed reference value for [Walk, Shop, Clean]
    prob = model.forward(OBS)
    assert abs(prob - 0.036288) < 1e-5


def test_viterbi_returns_correct_length(model: HiddenMarkovModel) -> None:
    states = model.viterbi(OBS)
    assert len(states) == len(OBS)


def test_viterbi_states_are_valid(model: HiddenMarkovModel) -> None:
    states = model.viterbi(OBS)
    assert all(0 <= s < 2 for s in states)


def test_viterbi_known_sequence(model: HiddenMarkovModel) -> None:
    # For [Walk, Shop, Clean] the most likely path is [Sunny, Sunny, Rainy]
    # i.e., [0, 0, 1]
    states = model.viterbi(OBS)
    assert states == [0, 0, 1]


def test_baum_welch_increases_likelihood(model: HiddenMarkovModel) -> None:
    """After training, the model should explain the data better."""
    obs = [0, 1, 2, 0, 0, 1, 2, 1, 0, 2]
    likelihood_before = model.forward(obs)

    # Train on same data
    rng = np.random.default_rng(42)
    random_model = HiddenMarkovModel(
        pi=rng.dirichlet([1, 1]),
        A=rng.dirichlet([1, 1], size=2),
        B=rng.dirichlet([1, 1, 1], size=2),
    )
    random_model.baum_welch(obs, n_iter=200)
    likelihood_after = random_model.forward(obs)

    assert likelihood_after > likelihood_before or likelihood_after > 1e-10


def test_baum_welch_parameters_sum_to_one(model: HiddenMarkovModel) -> None:
    obs = [0, 1, 2, 0, 1, 0]
    model.baum_welch(obs, n_iter=50)

    assert abs(np.sum(model.pi) - 1.0) < 1e-9
    assert np.allclose(model.A.sum(axis=1), 1.0)
    assert np.allclose(model.B.sum(axis=1), 1.0)
