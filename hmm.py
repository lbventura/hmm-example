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
