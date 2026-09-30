Modify the existing first-order autoregressive language model into a second-order autoregressive language model.

The probabilistic model must estimate:

P(Xt | Xt-2, Xt-1)

Requirements:

1. Replace pairwise transition counts with counts of observed token triples.

2. Store counts for contexts of the form:

   (Xt-2, Xt-1) -> Xt

3. Construct conditional probability tables using:

   P(Xt | Xt-2, Xt-1)

   ## = Count(Xt-2, Xt-1, Xt)

   Count(Xt-2, Xt-1)

4. Implement:

   * Training
   * Probability table construction
   * Next-word prediction
   * Sentence generation
   * Probability-normalization testing

5. Generation:

   * Begin with <START>.
   * Use the previous two tokens to predict the next token.
   * Continue until <END> is generated.

6. Reporting:

   * Print the number of distinct contexts.
   * Print the probability distribution for selected contexts.
   * Generate at least 20 example sentences.

7. Constraints:

   * Use ordinary Python data structures.
   * Do not use machine-learning libraries.
   * Do not use neural networks.
   * Do not use pretrained language models.

Include comments explaining how triple counts are converted into conditional probabilities.
