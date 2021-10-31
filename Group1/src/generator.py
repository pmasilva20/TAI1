import numpy as np
from fcm import FCM
from utils import read_text
import pprint

class Generator:

    def __init__(self, model, k , text_size, prior, alpha=None):
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
        # self.alpha = alpha if alpha else set(model.words)
        self.text_size = text_size
        self.prior = prior

    
    def gen_next_token(self, context):

        """
            calculate the next char of the generated text based on a prior context
        """

        # if the context doesnt exist in the table, then return a random char from the alphabet
        if context not in self.model:
            print("aki")
            return np.random.choice(self.alpha , 1)[0]

        n_char, prob_char = map(list, zip(*self.model[context]))
                
        return np.random.choice(n_char, 1, p=prob_char)[0]

    def gen_text(self):

        """
            Generate text based on fc model
        """
        
        assert len(self.prior) == self.k , 'First word must have the same size of context ('+ str(self.k) + ')'

        gen_text = list(self.prior)

        for i in range(self.text_size):

            gen_text.append(self.gen_next_token(''.join(gen_text[-self.k:])))
 
        return ''.join(gen_text)
    




def main():

    pp = pprint.PrettyPrinter(indent=4)

    a = 0
    k = 3

    prior = 'the'

    text = read_text('../example/example.txt')

    fcm = FCM(text,a,k)


    prob_dic = fcm.calculate_probabilities()

    gen =  Generator(fcm.prob_dic, fcm.k, 20, prior, list(set(fcm.words)))

    print(gen.gen_text())



if __name__ == "__main__":
    main()

