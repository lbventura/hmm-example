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
