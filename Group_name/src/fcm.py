import re
import math
import random
import sys

class FCM:
    
    def __init__(self, text,a=0,k=1):
        self.words = text
        self.fs_cache = {}
        self.a = a
        self.k = k
        self.contexts_seen = None
        self.prob_dic = None


    #Count all subsequences of k+1 words
    def count_subsequences(self,words,k):
        dic = {}
        for i in range(0,len(words),k+1):
            seq = tuple(words[i:i+(k+1)])
            if len(seq) < k+1:
                continue  
            if seq in dic:
                dic[seq] +=1
            else:
                dic[seq] = 1
        return dic

    #Get probability of a state c in the text
    def frequency_sequence(self,text,seq,k):
        t_seq = tuple(seq)
        if t_seq in self.fs_cache:
            return self.fs_cache[t_seq]
        total_seq = [text[i:i+k] for i in range(0,len(text),k+1)]
        remain_seq = [i for i in total_seq if i == seq ]

        result = len(remain_seq)/len(total_seq)
        self.fs_cache[t_seq] = result
        return result
    
    #Calculate all probabilities for each context
    def calculate_probabilities(self):
        unique_words = set(self.words)
        seq_count = self.count_subsequences(self.words,self.k)
        # Sequences Dictionary
        prob_dic = {}
        contexts_seen = {}
        for i in range(0,len(self.words),self.k+1):
            c = self.words[i:i+self.k]
            if tuple(c) in contexts_seen:
                continue
            #Get the sum of all contexts where c exists
            sum_all_constexts_c = sum([ seq_count[tuple(c + [simbol])] for simbol in unique_words if tuple(c + [simbol]) in seq_count])
            all_probs = []
            #Get all possible sequences for c + appending with another simbol
            for simbol in unique_words:
                seq = tuple(c + [simbol])
                if(seq not in seq_count):
                    continue
                n = seq_count[seq]
                total = (sum_all_constexts_c + self.a * len(unique_words))
                #Prob of event e for context c
                prob_n = (n + self.a)/total
                if tuple(c) in prob_dic:
                    prob_dic[tuple(c)].append((seq,prob_n))
                else:
                    prob_dic[tuple(c)] = [(seq,prob_n)]
                all_probs.append(prob_n)
            contexts_seen[tuple(c)] = all_probs
        self.contexts_seen = contexts_seen
        self.prob_dic = prob_dic
        return prob_dic

    #Calculate entropy taking into account 
    def calculate_entropy(self):
        context_entropy_list = []
        if self.contexts_seen == None: return 0
        for c in self.contexts_seen:
            all_probs = self.contexts_seen[c]
            #Get Hc entropy for each context/row
            entropy_of_context = -1 * sum([prob * math.log(prob,2) for prob in all_probs])
            #Probability of context/subsequence in total text
            prob_c = self.frequency_sequence(self.words,list(c),self.k)
            context_entropy_list.append(entropy_of_context * prob_c)
        return sum(context_entropy_list)
        

def read_text(address):
    try:
        with open(address,'r') as file:
            text_unfiltered = file.read()
            text_letters = list(text_unfiltered)
            return text_letters
    except:
        print("Error: No such a file or directory. Could not open/read file:", address)
        sys.exit()


def main():
    args = sys.argv[1:]
    if len(args) == 6 and args[0] == '-a' and args[2] == '-k' and args[4] == '-textpath':
        a= int(args[1])
        k=int(args[3])
        textpath=args[5]
        text = read_text(textpath)
        fcm = FCM(text,a,k)
        prob_dic = fcm.calculate_probabilities()
        entropy = fcm.calculate_entropy()
        for key in prob_dic:
            print(f'{key} : {prob_dic[key]}')
        print(f'Smoothing: {a} and Order: {k}')
        print(f'Entropy:{entropy}')
    else:
        print("Error: Bad use of Command Line Argument!")
        print("Usage:$ python3 fcm.py -a <smoothing_parameter> -k <order_of_the_model> -textpath <path_of_the_text_file> ")
        sys.exit()

if __name__ == "__main__":
    main()
