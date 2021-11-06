import numpy as np
from fcm import FCM
from utils import read_text
import pprint
import sys
import argparse


class Generator:

    def __init__(self, model, k, text_size, prior, alpha=None):
        """
            model - table resulting from statical information collection
            text_size - length of the text that will be generated
            prior - intial context to start the text generation
            alpha - model alphabet (list of all chars)
            k - context lenght
        """
        self.model = model
        self.k = k
        self.alpha = alpha
        self.text_size = text_size
        self.prior = prior

    def gen_next_token(self, context):

        """
            calculate the next char of the generated text based on a prior context
        """

        assert len(context) == self.k, 'Context must have size ' + self.k

        # if the context doesnt exist in the table, then return a random char from the alphabet
        if context not in self.model:
            return np.random.choice(self.alpha, 1)[0]

        n_char, prob_char = map(list, zip(*self.model[context]))

        return np.random.choice(n_char, 1, p=prob_char)[0]

    def gen_text(self):

        """
            Generate text based on fc model
        """

        assert len(self.prior) == self.k, 'First word must have the same size of context (' + str(self.k) + ')'

        gen_text = list(self.prior)

        for i in range(self.text_size):
            gen_text.append(self.gen_next_token(''.join(gen_text[-self.k:])))

        return ''.join(gen_text)


def main():
    pp = pprint.PrettyPrinter(indent=4)
    parser = argparse.ArgumentParser(description="Text Generator",
                                     usage="python3 generator.py -a <smoothing_parameter> -k <order_of_the_model> -path <path_of_the_text_file> -prior <initial_term> -s <gen_text_size>")

    parser.add_argument("-a", help="Smoothing parameter", type=float, required=True)
    parser.add_argument("-k", help="Model context size", type=int, required=True)
    parser.add_argument("-path", help="Path to text file", required=True)
    parser.add_argument("-p", "--prior", help="Prior", required=True)
    parser.add_argument("-s", "--size", help="Generated text size", type=int, default=40)

    args = parser.parse_args()

    if len(args.prior) != args.k:
        print("Error: Prior should have the same size of the order of the model!")
        sys.exit()
    text = read_text(args.path)
    fcm = FCM(text, args.a, args.k)
    fcm.calculate_probabilities()
    gen = Generator(fcm.prob_dic, fcm.k, args.size, args.prior, list(set(fcm.words)))
    print(gen.gen_text())


if __name__ == "__main__":
    main()
