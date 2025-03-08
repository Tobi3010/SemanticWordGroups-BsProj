from preprocessing import preprocessing
from datastorage import *
from network import word_network
from collections import defaultdict
import pandas as pd
import numpy as np
import time

def co_occurance(sentences, windowSize):
    vocab = set()
    dic = defaultdict(int)

    for s in sentences:                          
        for t1 in range(len(s)):                         # Loop all tokens in sentence

            if str.isdigit(s[t1]): continue              # If token is digit, ignore
            vocab.add(s[t1])                             # If not in set, add token to set, 

            for t2 in range(t1, t1 + windowSize):        # Loop tokens within window size
                if t2 >= len(s): break            
                if str.isdigit(s[t2]):                   # Digit represents stop words between two meaningful words, so skip
                    t2 += int(s[t2])                     # Skipping 
                    continue
                key = tuple(sorted([s[t2], s[t1]]))      # Make key of two tokens
                dic[key] += 1                            # Increment co-occurance count
                
    sorted(vocab)                                        # sort vocab
    return vocab, dic

def make_coo_matrix(vocab, dic):
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in dic.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df


def program():
    stopwords = load_stopwords("data/stop_words_english.txt")
    data_chunks = load_data("data/lyrics/lyrics.csv")

    for idx, chunk in enumerate(data_chunks):
        if idx == 0:
            df_chunks = pd.DataFrame(chunk)
            text = df_chunks.values.tolist()[0][0]
            sentences = preprocessing(text, stopwords)
            vocab, dic = co_occurance(sentences, 10)
        else:
            break
      
    df = make_coo_matrix(vocab, dic)
    df.to_csv("co_occurrence_matrix.csv", index=True)
    word_network()
    
start = time.time()  
program()
print("time taken : ", time.time() - start)








