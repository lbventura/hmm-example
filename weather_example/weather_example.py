"""Weather HMM demo.

Hidden states : Sunny (0), Rainy (1)
Observations  : Walk (0), Shop (1), Clean (2)
"""

import logging

import numpy as np

from .hmm import HiddenMarkovModel

logger = logging.getLogger(__name__)

# True model parameters (from Eisner 2002)
PI_TRUE: list[float] = [0.6, 0.4]

A_TRUE: list[list[float]] = [
    [0.7, 0.3],
    [0.4, 0.6],
]

B_TRUE: list[list[float]] = [
    [0.5, 0.4, 0.1],
    [0.1, 0.3, 0.6],
]

STATE_NAMES: list[str] = ["Sunny", "Rainy"]
OBS_NAMES: list[str] = ["Walk", "Shop", "Clean"]


def main() -> None:
    logger.info("  Hidden Markov Model — Weather Example")

    true_model = HiddenMarkovModel(PI_TRUE, A_TRUE, B_TRUE)

    obs, true_states = true_model.sample(length=10, rng=42)
    obs_names = [OBS_NAMES[o] for o in obs]
    state_names = [STATE_NAMES[s] for s in true_states]

    logger.info(f"\nGenerated observation sequence (length {len(obs)}):")
    logger.info("  " + " → ".join(obs_names))
    logger.info("\nHidden state sequence:")
    logger.info("  " + " → ".join(state_names))

    likelihood = true_model.forward(obs)
    logger.info("\n[Forward Algorithm]")
    logger.info(f"  P(observations | true model) = {likelihood:.8f}")

    decoded = true_model.viterbi(obs)
    decoded_names = [STATE_NAMES[s] for s in decoded]
    accuracy = sum(d == t for d, t in zip(decoded, true_states, strict=True)) / len(obs)

    logger.info("\n[Viterbi Algorithm]")
    logger.info("  Decoded : " + " → ".join(decoded_names))
    logger.info("  True    : " + " → ".join(state_names))
    logger.info(f"  Accuracy: {accuracy:.0%}")

    # Start from a random model and train on the observation sequence.
    rng = np.random.default_rng(7)
    random_model = HiddenMarkovModel(
        pi=rng.dirichlet([1, 1]),
        A=rng.dirichlet([1, 1], size=2),
        B=rng.dirichlet([1, 1, 1], size=2),
    )

    logger.info("\n[Baum-Welch Learning]")
    logger.info(f"  Initial likelihood: {random_model.forward(obs):.8f}")
    random_model.baum_welch(obs, n_iter=2000)
    logger.info(f"  Final   likelihood: {random_model.forward(obs):.8f}")

    logger.info("\n  Learned transition matrix A:")
    for i, row in enumerate(random_model.A):
        logger.info(f"    {STATE_NAMES[i]:6s}: " + "  ".join(f"{v:.3f}" for v in row))

    logger.info("\n  Learned emission matrix B:")
    for i, row in enumerate(random_model.B):
        logger.info(
            f"    {STATE_NAMES[i]:6s}: "
            + "  ".join(f"{OBS_NAMES[k]}={v:.3f}" for k, v in enumerate(row))
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
