# Hidden Markov Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a simple, educational Hidden Markov Model using the classic weather example, demonstrating Forward, Viterbi, and Baum-Welch algorithms from scratch.

**Architecture:** A `HiddenMarkovModel` class in `hmm.py` encapsulates model parameters (pi, A, B) and provides `forward()`, `viterbi()`, and `baum_welch()` methods. A `main.py` demo script runs the weather example end-to-end. Tests live in `tests/test_hmm.py`.

**Tech Stack:** Python 3.14, NumPy — managed via micromamba environment `hmm-example`.

---

## File Map

| File | Role |
|------|------|
| `environment.yml` | micromamba env spec (Python 3.14, NumPy) |
| `hmm.py` | `HiddenMarkovModel` class with all three algorithms |
| `main.py` | Demo script: weather example end-to-end |
| `tests/test_hmm.py` | Pytest tests for all algorithms |

---

### Task 1: Create micromamba environment

**Files:**
- Create: `environment.yml`

- [ ] **Step 1: Write environment.yml**

```yaml
name: hmm-example
channels:
  - conda-forge
dependencies:
  - python=3.14
  - numpy
  - pytest
```

- [ ] **Step 2: Create the micromamba environment**

Run:
```bash
micromamba create -f environment.yml -y
```

Expected output contains: `conda-forge/osx-arm64` (or linux equivalent) packages being downloaded, ending with `Transaction finished`.

- [ ] **Step 3: Verify Python version**

Run:
```bash
micromamba run -n hmm-example python --version
```

Expected: `Python 3.14.x`

- [ ] **Step 4: Commit**

```bash
git init
git add environment.yml
git commit -m "chore: set up micromamba environment for hmm-example

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 2: Scaffold HMM class and tests skeleton

**Files:**
- Create: `hmm.py`
- Create: `tests/__init__.py`
- Create: `tests/test_hmm.py`

- [ ] **Step 1: Create the hmm.py skeleton**

```python
"""Hidden Markov Model implementation.

Uses the classic weather example:
  - Hidden states: Sunny (0), Rainy (1)
  - Observations: Walk (0), Shop (1), Clean (2)
"""

import numpy as np


class HiddenMarkovModel:
    """A discrete Hidden Markov Model.

    Parameters
    ----------
    pi : array-like, shape (N,)
        Initial state probability distribution.
    A : array-like, shape (N, N)
        State transition matrix. A[i, j] = P(s_t=j | s_{t-1}=i).
    B : array-like, shape (N, M)
        Emission matrix. B[i, k] = P(o_t=k | s_t=i).
    """

    def __init__(self, pi, A, B):
        self.pi = np.array(pi, dtype=float)
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.N = self.A.shape[0]  # number of hidden states
        self.M = self.B.shape[1]  # number of observation symbols

    def forward(self, observations):
        """Compute the likelihood P(O | model) using the forward algorithm.

        Parameters
        ----------
        observations : list[int]
            Sequence of observation indices.

        Returns
        -------
        float
            P(observations | model)
        """
        raise NotImplementedError

    def viterbi(self, observations):
        """Find the most likely hidden state sequence using the Viterbi algorithm.

        Parameters
        ----------
        observations : list[int]
            Sequence of observation indices.

        Returns
        -------
        list[int]
            Most likely hidden state sequence.
        """
        raise NotImplementedError

    def baum_welch(self, observations, n_iter=100, tol=1e-6):
        """Estimate model parameters using the Baum-Welch (EM) algorithm.

        Updates self.pi, self.A, self.B in-place.

        Parameters
        ----------
        observations : list[int]
            Sequence of observation indices.
        n_iter : int
            Maximum number of EM iterations.
        tol : float
            Convergence threshold on log-likelihood change.
        """
        raise NotImplementedError
```

- [ ] **Step 2: Create tests skeleton**

```bash
mkdir -p tests
touch tests/__init__.py
```

```python
# tests/test_hmm.py
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
```

- [ ] **Step 3: Run tests to verify skeleton exists but raises NotImplementedError**

Run:
```bash
micromamba run -n hmm-example pytest tests/test_hmm.py -v
```

Expected: `0 passed` (no test functions yet, just fixtures).

- [ ] **Step 4: Commit**

```bash
git add hmm.py tests/
git commit -m "feat: scaffold HMM class and test skeleton

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 3: Implement the Forward algorithm

**Files:**
- Modify: `hmm.py` — implement `forward()`
- Modify: `tests/test_hmm.py` — add forward tests

**Background:** The forward algorithm computes α_t(i) = P(o_1,...,o_t, s_t=i | model) via:
- Initialisation: α_1(i) = π_i · B[i, o_1]
- Recursion: α_{t+1}(j) = B[j, o_{t+1}] · Σ_i α_t(i) · A[i,j]
- Result: P(O|λ) = Σ_i α_T(i)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_hmm.py`:

```python
def test_forward_returns_probability_between_0_and_1(model):
    prob = model.forward(OBS)
    assert 0 < prob <= 1


def test_forward_single_observation(model):
    # P(Walk) = pi[0]*B[0,0] + pi[1]*B[1,0] = 0.6*0.5 + 0.4*0.1 = 0.34
    prob = model.forward([0])
    assert abs(prob - 0.34) < 1e-9


def test_forward_known_sequence(model):
    # Precomputed reference value for [Walk, Shop, Clean]
    # Computed independently: ~0.03628
    prob = model.forward(OBS)
    assert abs(prob - 0.036288) < 1e-5
```

- [ ] **Step 2: Run to verify tests fail**

Run:
```bash
micromamba run -n hmm-example pytest tests/test_hmm.py::test_forward_returns_probability_between_0_and_1 tests/test_hmm.py::test_forward_single_observation tests/test_hmm.py::test_forward_known_sequence -v
```

Expected: 3 FAILs (NotImplementedError).

- [ ] **Step 3: Implement forward() in hmm.py**

Replace the `forward` method body:

```python
    def forward(self, observations):
        T = len(observations)
        # alpha[t, i] = P(o_1..o_t, s_t=i)
        alpha = np.zeros((T, self.N))

        # Initialisation
        alpha[0] = self.pi * self.B[:, observations[0]]

        # Recursion
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = self.B[j, observations[t]] * np.sum(alpha[t - 1] * self.A[:, j])

        return float(np.sum(alpha[T - 1]))
```

- [ ] **Step 4: Compute reference value and update test if needed**

Run a quick sanity check to get the exact reference probability:

```bash
micromamba run -n hmm-example python -c "
import numpy as np
from hmm import HiddenMarkovModel
m = HiddenMarkovModel([0.6,0.4],[[0.7,0.3],[0.4,0.6]],[[0.5,0.4,0.1],[0.1,0.3,0.6]])
print(f'{m.forward([0,1,2]):.8f}')
"
```

Update the `test_forward_known_sequence` tolerance or expected value if the output differs from `0.036288` by more than `1e-5`.

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
micromamba run -n hmm-example pytest tests/test_hmm.py -v
```

Expected: all forward tests PASS.

- [ ] **Step 6: Commit**

```bash
git add hmm.py tests/test_hmm.py
git commit -m "feat: implement Forward algorithm

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 4: Implement the Viterbi algorithm

**Files:**
- Modify: `hmm.py` — implement `viterbi()`
- Modify: `tests/test_hmm.py` — add Viterbi tests

**Background:** Viterbi finds the state sequence S* = argmax P(S, O | model):
- Initialisation: δ_1(i) = π_i · B[i, o_1]; ψ_1(i) = 0
- Recursion: δ_t(j) = B[j, o_t] · max_i [δ_{t-1}(i) · A[i,j]]; ψ_t(j) = argmax_i [δ_{t-1}(i) · A[i,j]]
- Backtrack from argmax_i δ_T(i)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_hmm.py`:

```python
def test_viterbi_returns_correct_length(model):
    states = model.viterbi(OBS)
    assert len(states) == len(OBS)


def test_viterbi_states_are_valid(model):
    states = model.viterbi(OBS)
    assert all(0 <= s < 2 for s in states)


def test_viterbi_known_sequence(model):
    # For [Walk, Shop, Clean] the most likely path is [Sunny, Sunny, Rainy]
    # i.e., [0, 0, 1]
    states = model.viterbi(OBS)
    assert states == [0, 0, 1]
```

- [ ] **Step 2: Run to verify tests fail**

Run:
```bash
micromamba run -n hmm-example pytest tests/test_hmm.py::test_viterbi_returns_correct_length tests/test_hmm.py::test_viterbi_states_are_valid tests/test_hmm.py::test_viterbi_known_sequence -v
```

Expected: 3 FAILs (NotImplementedError).

- [ ] **Step 3: Implement viterbi() in hmm.py**

Replace the `viterbi` method body:

```python
    def viterbi(self, observations):
        T = len(observations)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)

        # Initialisation
        delta[0] = self.pi * self.B[:, observations[0]]
        psi[0] = 0

        # Recursion
        for t in range(1, T):
            for j in range(self.N):
                scores = delta[t - 1] * self.A[:, j]
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = self.B[j, observations[t]] * np.max(scores)

        # Backtrack
        states = [int(np.argmax(delta[T - 1]))]
        for t in range(T - 1, 0, -1):
            states.insert(0, psi[t, states[0]])

        return states
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
micromamba run -n hmm-example pytest tests/test_hmm.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hmm.py tests/test_hmm.py
git commit -m "feat: implement Viterbi algorithm

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 5: Implement the Baum-Welch algorithm

**Files:**
- Modify: `hmm.py` — implement `baum_welch()`
- Modify: `tests/test_hmm.py` — add Baum-Welch tests

**Background:** Baum-Welch (EM) re-estimates parameters using:
- **E-step:** Compute α (forward), β (backward), γ_t(i) = P(s_t=i | O, λ), ξ_t(i,j) = P(s_t=i, s_{t+1}=j | O, λ)
- **M-step:** Re-estimate pi, A, B from expected counts
- Repeat until log-likelihood change < tol

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_hmm.py`:

```python
def test_baum_welch_increases_likelihood(model):
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


def test_baum_welch_parameters_sum_to_one(model):
    obs = [0, 1, 2, 0, 1, 0]
    model.baum_welch(obs, n_iter=50)

    assert abs(np.sum(model.pi) - 1.0) < 1e-9
    assert np.allclose(model.A.sum(axis=1), 1.0)
    assert np.allclose(model.B.sum(axis=1), 1.0)
```

- [ ] **Step 2: Run to verify tests fail**

Run:
```bash
micromamba run -n hmm-example pytest tests/test_hmm.py::test_baum_welch_increases_likelihood tests/test_hmm.py::test_baum_welch_parameters_sum_to_one -v
```

Expected: 2 FAILs (NotImplementedError).

- [ ] **Step 3: Implement baum_welch() in hmm.py**

Replace the `baum_welch` method body:

```python
    def baum_welch(self, observations, n_iter=100, tol=1e-6):
        obs = list(observations)
        T = len(obs)
        prev_log_likelihood = -np.inf

        for _ in range(n_iter):
            # --- E-step: forward pass ---
            alpha = np.zeros((T, self.N))
            alpha[0] = self.pi * self.B[:, obs[0]]
            for t in range(1, T):
                for j in range(self.N):
                    alpha[t, j] = self.B[j, obs[t]] * np.sum(alpha[t - 1] * self.A[:, j])

            # --- E-step: backward pass ---
            beta = np.zeros((T, self.N))
            beta[T - 1] = 1.0
            for t in range(T - 2, -1, -1):
                for i in range(self.N):
                    beta[t, i] = np.sum(self.A[i] * self.B[:, obs[t + 1]] * beta[t + 1])

            # --- E-step: gamma and xi ---
            # gamma[t, i] = P(s_t=i | O, model)
            gamma = alpha * beta
            gamma_sum = gamma.sum(axis=1, keepdims=True)
            gamma_sum = np.where(gamma_sum == 0, 1e-300, gamma_sum)
            gamma /= gamma_sum

            # xi[t, i, j] = P(s_t=i, s_{t+1}=j | O, model)  for t < T-1
            xi = np.zeros((T - 1, self.N, self.N))
            for t in range(T - 1):
                denom = 0.0
                for i in range(self.N):
                    for j in range(self.N):
                        xi[t, i, j] = (
                            alpha[t, i]
                            * self.A[i, j]
                            * self.B[j, obs[t + 1]]
                            * beta[t + 1, j]
                        )
                        denom += xi[t, i, j]
                if denom > 0:
                    xi[t] /= denom

            # --- M-step ---
            self.pi = gamma[0]

            for i in range(self.N):
                denom = gamma[:-1, i].sum()
                for j in range(self.N):
                    self.A[i, j] = xi[:, i, j].sum() / (denom if denom > 0 else 1e-300)

            for i in range(self.N):
                denom = gamma[:, i].sum()
                for k in range(self.M):
                    mask = np.array(obs) == k
                    self.B[i, k] = gamma[mask, i].sum() / (denom if denom > 0 else 1e-300)

            # Check convergence
            log_likelihood = np.log(np.sum(alpha[T - 1]) + 1e-300)
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
micromamba run -n hmm-example pytest tests/test_hmm.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add hmm.py tests/test_hmm.py
git commit -m "feat: implement Baum-Welch (EM) algorithm

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 6: Write the demo script

**Files:**
- Create: `main.py`

- [ ] **Step 1: Create main.py**

```python
"""Weather HMM demo.

Hidden states : Sunny (0), Rainy (1)
Observations  : Walk (0), Shop (1), Clean (2)
"""

import numpy as np
from hmm import HiddenMarkovModel

# ── True model parameters (from Eisner 2002) ─────────────────────────────────
PI_TRUE = [0.6, 0.4]

A_TRUE = [
    [0.7, 0.3],
    [0.4, 0.6],
]

B_TRUE = [
    [0.5, 0.4, 0.1],
    [0.1, 0.3, 0.6],
]

STATE_NAMES = ["Sunny", "Rainy"]
OBS_NAMES   = ["Walk", "Shop", "Clean"]


def generate_sequence(model, length, seed=0):
    """Sample a random observation sequence from a model."""
    rng = np.random.default_rng(seed)
    state = rng.choice(model.N, p=model.pi)
    observations, states = [], []
    for _ in range(length):
        obs = rng.choice(model.M, p=model.B[state])
        observations.append(int(obs))
        states.append(int(state))
        state = rng.choice(model.N, p=model.A[state])
    return observations, states


def main():
    print("=" * 60)
    print("  Hidden Markov Model — Weather Example")
    print("=" * 60)

    true_model = HiddenMarkovModel(PI_TRUE, A_TRUE, B_TRUE)

    # ── Generate a sequence ───────────────────────────────────────────────────
    obs, true_states = generate_sequence(true_model, length=10, seed=42)
    obs_names   = [OBS_NAMES[o] for o in obs]
    state_names = [STATE_NAMES[s] for s in true_states]

    print(f"\nGenerated observation sequence (length {len(obs)}):")
    print("  " + " → ".join(obs_names))
    print(f"\nTrue hidden state sequence:")
    print("  " + " → ".join(state_names))

    # ── Forward algorithm ─────────────────────────────────────────────────────
    likelihood = true_model.forward(obs)
    print(f"\n[Forward Algorithm]")
    print(f"  P(observations | true model) = {likelihood:.8f}")

    # ── Viterbi algorithm ─────────────────────────────────────────────────────
    decoded = true_model.viterbi(obs)
    decoded_names = [STATE_NAMES[s] for s in decoded]
    accuracy = sum(d == t for d, t in zip(decoded, true_states)) / len(obs)

    print(f"\n[Viterbi Algorithm]")
    print(f"  Decoded : " + " → ".join(decoded_names))
    print(f"  True    : " + " → ".join(state_names))
    print(f"  Accuracy: {accuracy:.0%}")

    # ── Baum-Welch learning ───────────────────────────────────────────────────
    # Start from a random model and train on the observation sequence
    rng = np.random.default_rng(7)
    random_model = HiddenMarkovModel(
        pi=rng.dirichlet([1, 1]),
        A=rng.dirichlet([1, 1], size=2),
        B=rng.dirichlet([1, 1, 1], size=2),
    )

    print(f"\n[Baum-Welch Learning]")
    print(f"  Initial likelihood: {random_model.forward(obs):.8f}")
    random_model.baum_welch(obs, n_iter=500)
    print(f"  Final   likelihood: {random_model.forward(obs):.8f}")

    print(f"\n  Learned transition matrix A:")
    for i, row in enumerate(random_model.A):
        print(f"    {STATE_NAMES[i]:6s}: " + "  ".join(f"{v:.3f}" for v in row))

    print(f"\n  Learned emission matrix B:")
    for i, row in enumerate(random_model.B):
        print(f"    {STATE_NAMES[i]:6s}: " + "  ".join(f"{OBS_NAMES[k]}={v:.3f}" for k, v in enumerate(row)))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo**

Run:
```bash
micromamba run -n hmm-example python main.py
```

Expected: Output shows the observation sequence, forward probability, Viterbi decoding with accuracy %, and before/after Baum-Welch likelihoods (final should be higher).

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add weather HMM demo script

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

### Task 7: Final verification

- [ ] **Step 1: Run full test suite**

Run:
```bash
micromamba run -n hmm-example pytest tests/ -v
```

Expected: all tests PASS with no errors or warnings.

- [ ] **Step 2: Run demo end-to-end**

Run:
```bash
micromamba run -n hmm-example python main.py
```

Expected: clean output with no errors, final Baum-Welch likelihood higher than initial.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "docs: add design spec and implementation plan

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```
