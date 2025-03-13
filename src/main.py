from preprocessing import preprocessing
from datastorage import *
from network import word_network
from collections import defaultdict
import pandas as pd
import numpy as np
import time
from itertools import islice
import os

def co_occurance(sentences, windowSize, vocab, dic):
    for s in sentences:                          
        for t1 in range(len(s)):                            # Loop all tokens in sentence
            if str.isdigit(s[t1]): continue                 # If token is digit, ignore
            vocab[s[t1]] += 1                               # Count the occurances of tokens, 

            t2 = t1 + 1
            while t2 < len(s) and t2 < t1+1+windowSize:     # Loop through tokens within windowsize
                if str.isdigit(s[t2]):                      # Digit represent stop words between meaningfull words, so skip
                    t2 += int(s[t2])                        # skipping
                    continue
                key = tuple(sorted([s[t2], s[t1]]))         # Make key of co-occurrance pair
                dic[key] += 1                               # Increment co-occurance count
                t2 += 1                                     
                                                   
    return vocab, dic



def pmi(vocab, dic):
    total = sum(vocab.values())  # Sum the total frequency
    for key in dic:
        # Compute PMI and update the value in dic
        dic[key] = np.max(np.log2(dic[key]/total / ((vocab[key[0]] / total) * (vocab[key[1]] / total))), 0)
    
    return vocab, dic


def make_co_matrix(vocab, dic):
    vocab_list = list(vocab.keys())  # Ensure only top 500 words are used
    df = pd.DataFrame(data=np.zeros((len(vocab_list), len(vocab_list)), dtype=np.float64),
                      index=vocab_list,
                      columns=vocab_list)
    
    for key, value in dic.items():
        if key[0] in vocab_list and key[1] in vocab_list:  # Extra safety check
            df.at[key[0], key[1]] = value
            df.at[key[1], key[0]] = value

    return df


def program():
    data_path = "data/books/SFGram-dataset"
    stopwords = load_stopwords("data/stop_words_english.txt")
    dic = defaultdict(int)
    vocab = defaultdict(int)

    os.chdir("data/books/SFGram-dataset")
    for file in os.listdir():
        print(f"Processing book : {file}")
        if (file == "book00005.txt" or file == "book00017.txt") : continue
        if (file == "book00026.txt"): break
        data_chunks = load_book_data(f"{file}")
        for idx, chunk in enumerate(data_chunks):
            df_chunks = pd.DataFrame(chunk)
            for row in df_chunks.values.tolist():
                text = row[0]
                sentences = preprocessing(text, stopwords)
                vocab, dic = co_occurance(sentences, 3, vocab, dic)
    os.chdir("../../../")
    
    print("Sorting data")
    vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))
    vocab = dict(islice(vocab.items(), 100))
    dic = {k: v for k, v in dic.items() if k[0] in vocab and k[1] in vocab}

    print("Creating Matrix")
    vocab, dic = pmi(vocab, dic)
    df = make_co_matrix(vocab, dic)
    
    print(df)

    print("Printing Matrix to .CSV")
    df.to_csv("co_occurrence_matrix.csv", index=True)

    print("Creating Word Network")
    word_network()
    
start = time.time()  
program()
print("time taken : ", time.time() - start)







def test_pmi():
    # Example co-occurrence matrix
 
    dic = {("foo", "bar") : 3}
    vocab = {"foo" : 3, "bar" : 8}

    vocab, dic = pmi(vocab, dic)
    computed_pmi = dic[("foo", "bar")]
  
    total = 23  
    P_foo_bar = 3 / total
    P_foo = 3 / total
    P_bar = 8 / total
    expected_pmi = np.log2(P_foo_bar / (P_foo * P_bar))

    print(f"Expected PMI(foo, bar): {expected_pmi:.6f}")
    print(f"Computed PMI(foo, bar): {computed_pmi:.6f}")
    assert np.isclose(computed_pmi, expected_pmi, atol=1e-5), "PMI calculation is incorrect!"

    print("PMI test passed! ✅")
