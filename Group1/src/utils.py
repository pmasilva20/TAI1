

def read_text(address):
    with open(address,'r') as file:
        text_unfiltered = file.read()
        text_letters = list(text_unfiltered)
        return text_letters
