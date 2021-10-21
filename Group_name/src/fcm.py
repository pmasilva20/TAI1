import re
import math
import random
import pprint
import cProfile
from utils import read_text


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
    def frequency_sequence(self,seq,total_len_seq):
        t_seq = tuple(seq)

        if t_seq in self.fs_cache:
            count = self.fs_cache[t_seq]
            result = count/total_len_seq
            return result
        return 0
    
    #Calculate all probabilities for each context
    def calculate_probabilities(self):
        unique_words = set(self.words)
        seq_count = self.count_subsequences(self.words,self.k)
        # Sequences Dictionary
        # for key in seq_count:
        #     print(f'{key} -> {seq_count[key]}')
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
        context_entropy_list = []
        if self.contexts_seen == None: return 0
        
        #Total number of sequences
        total_seq = [self.words[i:i+self.k] for i in range(0,len(self.words),self.k+1)]
        total_seq_len = len(total_seq)
        
        #Calculate frequencies before hand
        #Done in order to prevent multiple transversals of the whole text
        self.calculate_frequencies(total_seq)

        for c in self.contexts_seen:
            all_probs = self.contexts_seen[c]
            #Get Hc entropy for each context/row
            entropy_of_context = -1 * sum([prob * math.log(prob,2) for prob in all_probs])
            #Probability of context/subsequence in total text
            prob_c = self.frequency_sequence(list(c),total_seq_len)
            #print(f'For c={c} entropy={entropy_of_context} Cprob={prob_c}')
            context_entropy_list.append(entropy_of_context * prob_c)
        return sum(context_entropy_list)
        




#TODO:Check number of CPU's/cores available for PC,
#If 1 or None do all in one process, else divide
#list in N sublists and give each one to a process
#Might have to wait out others


def main():
    pp = pprint.PrettyPrinter(indent=4)

    a = 0
    k = 3
        
    text = read_text('../example/example.txt')

    fcm = FCM(text,a,k)
    prob_dic = fcm.calculate_probabilities()
    entropy = fcm.calculate_entropy()
    for key in prob_dic:
        pp.pprint(f'{key} : {prob_dic[key]}')
    pp.pprint(f'Smoothing: {a} and Order: {k}')
    pp.pprint(f'Entropy:{entropy}')


if __name__ == "__main__":
    #cProfile.run('main()',sort='tottime')
    main()
