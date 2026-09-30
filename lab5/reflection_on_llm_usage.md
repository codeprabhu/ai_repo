# Reflection on LLM Usage and Validation

The LLM was used as a programming assistant to generate implementations of both the first-order and second-order autoregressive language models. Instead of asking the LLM to directly write a language model, a detailed behavioural specification was provided describing the probabilistic model, the required probability distributions, the counting procedure, the generation mechanism, and the testing requirements.

For the first-order model, the LLM generated code that counted transitions between consecutive words and constructed conditional probability tables representing:

P(X_t | X_{t-1})

For the second-order model, the specification was modified to require estimation of:

P(X_t | X_{t-2}, X_{t-1})

using counts of observed token triples.

The generated implementations were not accepted blindly. The code was inspected to verify:

- where transition counts were stored,
- how conditional probabilities were computed,
- how next-word prediction was performed,
- how sentence generation was implemented.

The implementations were then validated experimentally. Probability-normalization tests were performed for every context, and all probability distributions summed to exactly 1.000000. This confirmed that the conditional probability tables were correctly constructed.

Generated sentences were also examined qualitatively. The first-order model occasionally produced less coherent sentences because it only considered one previous word. The second-order model produced more coherent outputs because it incorporated additional context from two previous words.

One modification made during validation was the addition of explicit probability-normalization tests and handling of unseen contexts during generation. These checks increased confidence that the implementation correctly represented the intended probabilistic model.

This exercise demonstrated the importance of treating the LLM as a tool for implementation rather than as a replacement for understanding. The probabilistic model was designed first, and the generated code was subsequently inspected and validated against the intended Bayesian-network specification.