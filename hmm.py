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
