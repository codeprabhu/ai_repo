from collections import defaultdict
import random


class SecondOrderLanguageModel:
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))
        self.probabilities = {}

    def train(self, sentences):

        for sentence in sentences:

            for i in range(len(sentence) - 2):

                context = (
                    sentence[i],
                    sentence[i + 1]
                )

                next_word = sentence[i + 2]

                self.counts[context][next_word] += 1

        self.probabilities = {}

        for context in self.counts:

            total = sum(
                self.counts[context].values()
            )

            self.probabilities[context] = {}

            for next_word in self.counts[context]:

                self.probabilities[context][next_word] = (
                    self.counts[context][next_word]
                    / total
                )

    def print_probability_table(self):

        print("\n=== Second Order CPT ===")

        for context in self.probabilities:

            print(f"\nContext {context}")

            for nxt, prob in self.probabilities[context].items():

                print(
                    f"  {nxt:<10} {prob:.4f}"
                )

    def predict_next(self, word1, word2):

        context = (word1, word2)

        if context not in self.probabilities:
            return None

        return max(
            self.probabilities[context],
            key=self.probabilities[context].get
        )

    def generate_sentence(self):

        w1 = "<START>"
        w2 = "the"

        sentence = ["the"]

        while True:

            context = (w1, w2)

            if context not in self.probabilities:
                break

            words = list(
                self.probabilities[context].keys()
            )

            probs = list(
                self.probabilities[context].values()
            )

            next_word = random.choices(
                words,
                weights=probs,
                k=1
            )[0]

            if next_word == "<END>":
                break

            sentence.append(next_word)

            w1 = w2
            w2 = next_word

        return " ".join(sentence)

    def test_normalization(self):

        print("\n=== Normalization Test ===")

        for context in self.probabilities:

            total = sum(
                self.probabilities[context].values()
            )

            print(
                f"{context} -> {total:.6f}"
            )

    def count_parameters(self):
        return sum(
            len(v)
            for v in self.probabilities.values()
        )


if __name__ == "__main__":

    dataset = [
        ["<START>", "the", "cat", "sat", "on", "the", "mat", "<END>"],
        ["<START>", "the", "cat", "sat", "on", "the", "rug", "<END>"],
        ["<START>", "the", "dog", "sat", "on", "the", "mat", "<END>"],
        ["<START>", "the", "dog", "ran", "to", "the", "park", "<END>"],
        ["<START>", "the", "cat", "ran", "to", "the", "park", "<END>"],
        ["<START>", "the", "dog", "sat", "on", "the", "rug", "<END>"]
    ]

    model = SecondOrderLanguageModel()

    model.train(dataset)

    model.print_probability_table()

    model.test_normalization()

    print("\nParameters:")
    print(model.count_parameters())

    print("\nPrediction after ('the','cat'):")
    print(model.predict_next("the", "cat"))

    print("\nGenerated Sentences")

    for i in range(20):
        print(f"{i+1}. {model.generate_sentence()}")