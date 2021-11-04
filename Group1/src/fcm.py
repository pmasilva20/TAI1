import re
import math
import random
import sys
import pprint
import cProfile
from utils import read_text
import time
import argparse

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
        for i in range(0,len(words)):
            seq = tuple(words[i:i+(k+1)])
            if len(seq) < k+1:
                continue  
            if seq in dic:
                dic[seq] +=1
            else:
                dic[seq] = 1
        return dic


    #Get probability of a state c in the text
    def frequency_sequence(self,seq,total_len_seq):
        t_seq = tuple(seq)

        if t_seq in self.fs_cache:
            count = self.fs_cache[t_seq]
            result = count/total_len_seq
            return result
        return 0
    
    #Calculate all probabilities for each context
    def calculate_probabilities(self):
        words_len = len(self.words)
        unique_words = set(self.words)
        seq_count = self.count_subsequences(self.words,self.k)
        # Sequences Dictionary
        # for key in seq_count:
        #     print(f'{key} -> {seq_count[key]}')
        prob_dic = {}
        contexts_seen = {}
        for i in range(0,words_len):
            c = self.words[i:i+self.k]
            if tuple(c) in contexts_seen:
                continue
            #Get the sum of all contexts where c exists
            sum_all_constexts_c = sum([ seq_count[tuple(c + [simbol])] for simbol in unique_words if tuple(c + [simbol]) in seq_count])
            all_probs = []
            #Get all possible sequences for c + appending with another simbol
            unique_words_len = len(unique_words)
            for simbol in unique_words:
                seq = tuple(c + [simbol])
                if(seq not in seq_count):
                    prob_n = 1/unique_words_len
                else:
                    n = seq_count[seq]
                    total = (sum_all_constexts_c + self.a * unique_words_len)
                    #Prob of event e for context c
                    prob_n = (n + self.a)/total

      
                #print(f'Seq:{seq} Nprob:{prob_n} the total from prob is {total}')
                context = ''.join(c)

                if context in prob_dic:
                    prob_dic[context].append((simbol,prob_n))
                else:
                    prob_dic[context] = [(simbol,prob_n)]
                all_probs.append(prob_n)
            contexts_seen[tuple(c)] = all_probs
        self.contexts_seen = contexts_seen
        self.prob_dic = prob_dic
        return prob_dic

    def calculate_frequencies(self,total_seq):
        for i in total_seq:
            seq_hash = tuple(i)
            if seq_hash in self.contexts_seen:
                if seq_hash in self.fs_cache:
                    self.fs_cache[seq_hash]+=1
                else:
                    self.fs_cache[seq_hash] = 0


    #Calculate entropy taking into account 
    def calculate_entropy(self):
        words_len = len(self.words)
        context_entropy_list = []
        if self.contexts_seen == None: return 0
        
        #Total number of sequences
        total_seq = [self.words[i:i+self.k] for i in range(0,words_len)]
        total_seq_len = len(total_seq)
        
        #Calculate frequencies before hand
        #Done in order to prevent multiple transversals of the whole text
        self.calculate_frequencies(total_seq)

        for c in self.contexts_seen:
            all_probs = self.contexts_seen[c]
            #Get Hc entropy for each context/row
            #He gets 0 if prob is 0
            entropy_of_context = -1 * sum([prob * math.log(prob,2) for prob in all_probs])
            #Probability of context/subsequence in total text
            prob_c = self.frequency_sequence(list(c),total_seq_len)
            #print(f'For c={c} entropy={entropy_of_context} Cprob={prob_c}')
            context_entropy_list.append(entropy_of_context * prob_c)
        return sum(context_entropy_list)
        

def main():
    pp = pprint.PrettyPrinter(indent=4)
    parser = argparse.ArgumentParser(description= "Calculate Entropy",
    usage="python3 fcm.py -a <smoothing_parameter> -k <order_of_the_model> -path <path_of_the_text_file>")
    
    parser.add_argument("-a", help= "Smoothing parameter", type=float, required=True)
    parser.add_argument("-k", help= "Model context size",type=int, required=True)
    parser.add_argument("-path", help= "Path to text file", required=True)

    args = parser.parse_args()

    text = read_text(args.path)
    fcm = FCM(text,args.a,args.k)
    prob_dic = fcm.calculate_probabilities()
    entropy = fcm.calculate_entropy()
    # for key in prob_dic:
    #     pp.pprint(f'{key} : {prob_dic[key]}')
    pp.pprint(f'Smoothing: {args.a} and Order: {args.k}')
    pp.pprint(f'Entropy:{entropy}')


if __name__ == "__main__":
    main()
