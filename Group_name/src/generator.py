import numpy as np



class Generator:

    def __init__(self, model, text_size, prior):
        """
            model - fcm resulting from statical information collection
            text_size - length of the text that will be generated
            initial_word - intial context to start the text generation
        """
        self.model = model
        self.text_size = text_size
        self.prior = prior

    
    def gen_next_token(self, context):

        return np.random.choice(self.model.proprob_dic[context])
    

    def gen_text(self):
        
        assert self.prior >= self.model.k , 'First word must have at least size '+ self.model.k

        gen_text = self.prior

        gen_text += self.gen_next_token(gen_text[-self.model.k:])

        return gen_text
    





