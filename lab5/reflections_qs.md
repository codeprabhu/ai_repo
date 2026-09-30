# Bayesian Networks and Autoregressive Language Models – Answers

## Question 1

### Why is the chain-rule decomposition useful for generating text?

The chain rule decomposes the probability of an entire sentence into a sequence of conditional probabilities. Instead of estimating the probability of a complete sentence directly, the model only needs to estimate the probability of the next word given the previous words.

$$
P(X_1,\ldots,X_T)=P(X_1)\prod_{t=2}^{T}P(X_t|X_1,\ldots,X_{t-1})
$$

This allows text generation to proceed one word at a time. At each step, the model predicts the next word using the previously generated words and continues until an end token is produced.

---

## Question 2

### What independence assumption is being made by the first-order Bayesian network?

The first-order model assumes that each word depends only on the immediately preceding word.

$$
P(X_t|X_1,\ldots,X_{t-1})
=
P(X_t|X_{t-1})
$$

For example,

$$
P(X_4|X_1,X_2,X_3)
=
P(X_4|X_3)
$$

This is known as the first-order Markov assumption.

---

## Question 3

### Construct the conditional probability distribution \(P(\text{next word}|\text{current word})\)

Training dataset:

```text
<START> the cat sat on the mat <END>
<START> the cat sat on the rug <END>
<START> the dog sat on the mat <END>
<START> the dog ran to the park <END>
<START> the cat ran to the park <END>
<START> the dog sat on the rug <END>
```

### Current Word: the

| Next Word | Count | Probability |
| --------- | ----- | ----------- |
| cat       | 3     | 0.25        |
| dog       | 3     | 0.25        |
| mat       | 2     | 0.167       |
| rug       | 2     | 0.167       |
| park      | 2     | 0.167       |

### Current Word: cat

| Next Word | Count | Probability |
| --------- | ----- | ----------- |
| sat       | 2     | 0.667       |
| ran       | 1     | 0.333       |

### Current Word: dog

| Next Word | Count | Probability |
| --------- | ----- | ----------- |
| sat       | 2     | 0.667       |
| ran       | 1     | 0.333       |

### Current Word: sat

| Next Word | Count | Probability |
| --------- | ----- | ----------- |
| on        | 4     | 1.0         |

### Current Word: ran

| Next Word | Count | Probability |
| --------- | ----- | ----------- |
| to        | 2     | 1.0         |

### Zero-Probability Transitions

Examples:

* \(P(ran|sat)=0\)
* \(P(cat|dog)=0\)
* \(P(mat|cat)=0\)
* \(P(dog|ran)=0\)

These transitions never occur in the training dataset.

---

## Question 4

### Where in the program are the transition counts stored?

The transition counts are stored in a dictionary structure.

Example:

```python
counts[previous_word][next_word]
```

For example:

```python
counts["the"]["cat"] = 3
```

stores the fact that the word "cat" follows "the" three times.

---

## Question 5

### Where is \(P(X_t|X_{t-1})\) computed?

The conditional probabilities are computed by dividing transition counts by the total number of outgoing transitions from a word.

Example:

```python
for prev in counts:
    total = sum(counts[prev].values())

    for nxt in counts[prev]:
        probabilities[prev][nxt] = (
            counts[prev][nxt] / total
        )
```

This implements

$$
P(w_j|w_i)
=
\frac{C(w_i,w_j)}
{\sum_k C(w_i,w_k)}
$$

---

## Question 6

### How does the program choose the next word?

There are two possible approaches.

#### Greedy Selection

Always chooses the most probable next word.

```python
max(probabilities[current_word])
```

#### Probabilistic Sampling

Samples from the probability distribution.

```python
random.choices(words, weights=probs)
```

Greedy selection always produces the same output for a given context, while sampling can generate different outputs according to the learned probabilities.

---

## Question 7

### What happens if the program encounters a word for which no transition has been observed?

The model cannot predict a next word because no transition information exists.

Possible solutions include:

* Returning `<END>`
* Restarting generation
* Applying smoothing techniques

Without handling this case, the program may terminate unexpectedly.

---

## Question 8

### If one of the totals is 0.87, what does this tell you?

The implementation is incorrect.

For a valid probability distribution,

$$
\sum_v P(v|w)=1
$$

must hold for every word \(w\).

A total of 0.87 indicates a normalization error or bug in the probability calculation.

---

# Question 9

## Are the most probable predictions always the same as the words you would personally expect?

Not necessarily.

For example, when the previous word is **"cat"**, the model predicts **"sat"** with probability **0.6667** and **"ran"** with probability **0.3333**. Similarly, after **"dog"**, the model predicts **"sat"** with probability **0.6667** and **"ran"** with probability **0.3333**.

These predictions are based entirely on the frequencies observed in the training dataset. Human expectations may also consider grammar, meaning, and real-world knowledge, whereas the model relies only on statistical evidence from the training data.

Therefore, the most probable prediction according to the model may not always match what a human would intuitively expect.

---

# Question 10

## Compare greedy generation and probabilistic sampling.

Greedy generation always selects the most probable next token at every step.

For example, using the learned probabilities, greedy generation would repeatedly favour transitions such as:

```text
the → cat
cat → sat
sat → on
on → the
the → mat
```

which tends to produce:

```text
the cat sat on the mat
```

Probabilistic sampling instead selects words according to their probabilities. In the generated outputs, the model produced a variety of sentences, including:

```text
the dog ran to the park
the cat ran to the park
the dog sat on the mat
the dog sat on the rug
the cat sat on the rug
```

Sampling produced significantly more variation because lower-probability transitions such as **cat → ran** and **dog → ran** could still be selected. Greedy generation prioritizes the most likely path, while sampling explores multiple possible paths through the Bayesian network.

---

# Question 11

## How does the second-order model differ from the first-order model?

| Aspect | First-Order Model | Second-Order Model |
|----------|----------|----------|
| Graph Structure | \(X_{t-1}\rightarrow X_t\) | \(X_{t-2}\rightarrow X_t \leftarrow X_{t-1}\) |
| CPT | \(P(X_t \mid X_{t-1})\) | \(P(X_t \mid X_{t-2},X_{t-1})\) |
| Context Available | One previous word | Two previous words |
| Number of Parameters | 14 transitions | 18 transitions |
| Data Requirement | Smaller | Larger |

The second-order model uses a larger conditional probability table because it must store probabilities for pairs of previous words instead of a single previous word.

---

# Question 12

## Why does increasing context improve prediction but make estimation harder?

The second-order model uses two previous words instead of one previous word when predicting the next token.

For example:

```text
(the, cat) → sat : 0.6667
(the, cat) → ran : 0.3333
```

whereas the first-order model only uses:

```text
cat → sat : 0.6667
cat → ran : 0.3333
```

Using additional context can improve prediction because the model has more information about the current state of the sentence.

However, increasing context also increases the size of the conditional probability table. In our implementation, the first-order model stored **14 transition parameters**, while the second-order model stored **18 parameters**.

As context size grows, many possible contexts may never appear in the training data, leading to sparse counts and zero-probability contexts. Therefore, larger models generally require more training data to estimate probabilities reliably.

---

# First-Order Model Results

## Probability Normalization Test

All probability distributions summed exactly to:

```text
1.000000
```

for every context. This confirms that the conditional probability tables were correctly normalized and represent valid probability distributions.

## Most Probable Prediction

```text
Prediction after "cat":
sat
```

## Sample Generated Sentences

```text
the dog ran to the cat sat on the park
the rug
the dog ran to the cat sat on the cat ran to the mat
the mat
the park
```

The generated sentences demonstrate the probabilistic nature of the model. Some outputs are grammatically unusual because the first-order model only considers one previous word.

---

# Second-Order Model Results

## Probability Normalization Test

All conditional probability distributions summed to:

```text
1.000000
```

for every two-word context.

## Number of Parameters

```text
18
```

## Most Probable Prediction

```text
Prediction after ("the", "cat"):
sat
```

## Sample Generated Sentences

```text
the cat sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug
the cat sat on the rug
```

The second-order model generated noticeably more coherent sentences because it used two previous words as context when predicting the next token.

## Question 13

### Why is Approach B preferable?

Approach B explicitly specifies:

* The intended probabilistic model
* The representation being used
* The desired behaviour
* The testing requirements

This makes it easier to verify whether the generated implementation actually corresponds to the intended model.

Approach A focuses only on obtaining code, whereas Approach B focuses on correctly implementing a specified intelligent system.

---

## Question 14

### What did thinking of the language model as a Bayesian network provide?

Viewing the language model as a Bayesian network provides:

1. A clear representation of dependencies between variables.
2. A factorization of the joint probability distribution.
3. A principled interpretation of conditional probabilities.
4. A systematic procedure for text generation.
5. A framework for reasoning about independence assumptions.
6. An understanding of how increasing context affects prediction.
7. A method for verifying whether an implementation matches its probabilistic specification.

The Bayesian network perspective makes the probabilistic structure of the language model explicit and easier to analyze.
