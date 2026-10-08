# Weather HMM example

A small discrete Hidden Markov Model with hidden states `Sunny` and `Rainy`, and observations `Walk`, `Shop`, and `Clean`. It illustrates sequence generation, inference, and learning from observations when the states are unknown.

## Modules and execution

- `hmm.py`: `HiddenMarkovModel`, with Forward likelihood calculation, Viterbi decoding, Baum-Welch training, and sequence sampling.
- `weather_example.py`: generates observations, decodes hidden states, and learns transition and emission probabilities from one short sequence.
- `compare_training.py`: compares basic and improved Baum-Welch training, aligning hidden-state labels before measuring the L1 error in the learned transition (`A`) and emission (`B`) matrices.

After following the [development setup](../README.md#set-up-a-development-environment), run these commands from the repository root:

```bash
python -m weather_example.weather_example
python -m weather_example.compare_training
python -m pytest tests/test_hmm.py
```

The training comparison uses more data for the improved method as well as different training settings; it does not isolate the effect of any one improvement.

## Improvements to Baum-Welch

The basic `HiddenMarkovModel.baum_welch` trains on one sequence using unscaled probabilities and no smoothing. The weather demo starts it from one random initialization. `compare_training.py` adds:

- **Multi-sequence training.** `baum_welch_improved` accumulates expected state occupancies and transition counts across independent sequences before updating parameters. More observations reduce sampling noise, though they do not guarantee recovery of the true parameters.
- **Multiple random restarts.** The comparison's `main` function calls the improved trainer from several initializations and keeps the model with the highest returned total training log-likelihood. Baum-Welch can converge to different local optima; restarts help explore them. The restarts are outside `baum_welch_improved` itself.
- **Additive Dirichlet smoothing.** The M-step updates each probability as `(count + pseudo) / (sum_of_counts + pseudo * n_categories)`, with `pseudo=0.5` by default. This is a MAP update under a symmetric `Dirichlet(pseudo + 1)` prior. Positive pseudocounts prevent zero-probability entries from becoming permanent and temper estimates from sparse counts.
- **Log-space Forward/Backward.** Probabilities can underflow as sequences grow longer. Log-space calculations with `logsumexp` improve numerical stability; they do not remove floating-point approximation or local optima.

Hidden-state labels are interchangeable: a learned state need not have the same index as the corresponding true state. The comparison permutes both axes of `A` and the rows of `B` together before calculating parameter error.

## Why Dirichlet initialization?

Both demos initialize the initial-state distribution (`pi`), each transition row (`A`), and each emission row (`B`) with draws from `Dirichlet(1, ..., 1)`:

- Each draw is a valid probability vector whose entries sum to one.
- Concentration one gives a uniform distribution over the probability simplex, without favoring a particular category.
- Random draws break the symmetry of identical rows and give restarts different starting points.
- Entries are strictly positive with probability one in the mathematical distribution, allowing log-space training to start without structural zeros.

Random initialization creates useful starting points, but does not guarantee convergence to the true model.
