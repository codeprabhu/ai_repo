You are implementing a probabilistic model, not a neural network.

Write a Python implementation of a first-order autoregressive language model based on transition counts.

Requirements:

1. Input:

   * Accept a list of tokenized sentences.
   * Each sentence contains the special tokens <START> and <END>.

2. Training:

   * Count transitions between consecutive tokens.
   * Store counts using standard Python data structures.
   * Estimate the conditional probability distribution

     P(Xt | Xt-1)

     using relative frequencies.

3. Probability Tables:

   * Construct and store conditional probability tables (CPTs).
   * Allow printing the probability distribution for any given token.

4. Prediction:

   * Implement a function that returns the most probable next token for a given token.

5. Generation:

   * Implement sentence generation by repeatedly sampling from the learned probability distribution.
   * Begin generation from <START>.
   * Stop when <END> is generated.

6. Testing:

   * Implement a function that verifies that for every token:

     Σ P(next_token | current_token) = 1

   * Print the total probability for every token.

7. Constraints:

   * Use only Python standard libraries.
   * Do not use machine-learning libraries.
   * Do not use neural networks.
   * Do not use pretrained language models.

8. Output:

   * Display transition counts.
   * Display conditional probability tables.
   * Demonstrate next-word prediction.
   * Generate at least 20 example sentences.

Include comments explaining how the transition counts are converted into conditional probabilities.
