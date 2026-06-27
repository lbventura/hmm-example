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
    [0.7, 0.3],   # Sunny -> Sunny=0.7, Sunny -> Rainy=0.3
    [0.4, 0.6],   # Rainy -> Sunny=0.4, Rainy -> Rainy=0.6
]

B = [
    [0.5, 0.4, 0.1],   # Sunny: Walk=0.5, Shop=0.4, Clean=0.1
    [0.1, 0.3, 0.6],   # Rainy: Walk=0.1, Shop=0.3, Clean=0.6
]

OBS = [0, 1, 2]  # Walk, Shop, Clean


@pytest.fixture
def model():
    return HiddenMarkovModel(PI, A, B)


def test_forward_returns_probability_between_0_and_1(model):
    prob = model.forward(OBS)
    assert 0 < prob <= 1


def test_forward_single_observation(model):
    # P(Walk) = pi[0]*B[0,0] + pi[1]*B[1,0] = 0.6*0.5 + 0.4*0.1 = 0.34
    prob = model.forward([0])
    assert abs(prob - 0.34) < 1e-9


def test_forward_known_sequence(model):
    # Precomputed reference value for [Walk, Shop, Clean]
    prob = model.forward(OBS)
    assert abs(prob - 0.036288) < 1e-5
