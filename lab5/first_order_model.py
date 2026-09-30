from collections import defaultdict
import random


class FirstOrderLanguageModel:
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))
        self.probabilities = {}

    def train(self, sentences):
        # Count transitions
        for sentence in sentences:
            for i in range(len(sentence) - 1):
                current_word = sentence[i]
                next_word = sentence[i + 1]
                self.counts[current_word][next_word] += 1

        # Convert counts to probabilities
        self.probabilities = {}

        for current_word in self.counts:
            total = sum(self.counts[current_word].values())

            self.probabilities[current_word] = {}

            for next_word in self.counts[current_word]:
                self.probabilities[current_word][next_word] = (
                    self.counts[current_word][next_word] / total
                )

    def print_transition_counts(self):
        print("\n=== Transition Counts ===")
        for word in self.counts:
            print(f"{word} -> {dict(self.counts[word])}")

    def print_probability_table(self):
        print("\n=== Conditional Probabilities ===")
        for word in self.probabilities:
            print(f"\nCurrent Word: {word}")

            for nxt, prob in self.probabilities[word].items():
                print(f"  {nxt:<10} {prob:.4f}")

    def get_distribution(self, word):
        return self.probabilities.get(word, {})

    def predict_next(self, word):
        if word not in self.probabilities:
            return None

        return max(
            self.probabilities[word],
            key=self.probabilities[word].get
        )

    def generate_sentence(self):
        current = "<START>"

        sentence = []

        while True:

            if current not in self.probabilities:
                break

            words = list(self.probabilities[current].keys())
            probs = list(self.probabilities[current].values())

            next_word = random.choices(
                words,
                weights=probs,
                k=1
            )[0]

            if next_word == "<END>":
                break

            sentence.append(next_word)

            current = next_word

        return " ".join(sentence)

    def test_normalization(self):
        print("\n=== Probability Normalization Test ===")

        for word in self.probabilities:
            total = sum(
                self.probabilities[word].values()
            )

            print(f"{word:<10} {total:.6f}")


if __name__ == "__main__":

    dataset = [
        ["<START>", "the", "cat", "sat", "on", "the", "mat", "<END>"],
        ["<START>", "the", "cat", "sat", "on", "the", "rug", "<END>"],
        ["<START>", "the", "dog", "sat", "on", "the", "mat", "<END>"],
        ["<START>", "the", "dog", "ran", "to", "the", "park", "<END>"],
        ["<START>", "the", "cat", "ran", "to", "the", "park", "<END>"],
        ["<START>", "the", "dog", "sat", "on", "the", "rug", "<END>"]
    ]

    model = FirstOrderLanguageModel()

    model.train(dataset)

    model.print_transition_counts()

    model.print_probability_table()

    model.test_normalization()

    print("\nPrediction after 'cat':")
    print(model.predict_next("cat"))

    print("\nGenerated Sentences")

    for i in range(20):
        print(f"{i+1}. {model.generate_sentence()}")