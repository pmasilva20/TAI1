import sys

def read_text(address):
    try:
        with open(address,'r') as file:
            text_unfiltered = file.read()
            text_letters = list(text_unfiltered)
            return text_letters
    except:
        print("Error: No such a file or directory. Could not open/read file:", address)
        sys.exit()