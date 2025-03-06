import csv
import nltk
import string
import pandas as pd
import numpy as np
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# Functions for data loading ----------------------------------------------------------------------------------
def load_data(filename):
    with open(filename, newline='', encoding='utf-8') as file:
        data = csv.DictReader(file)  # Automatically uses first row as headers
        lyrics_list = [row for row in data]  # Convert iterator to list of dictionaries

    return lyrics_list



# Functions for data preprocessing ----------------------------------------------------------------------------
def remove_uppercase(text):
    return text.lower()

def remove_numbers(text):
    text = re.sub(r'\d+', '', text)
    return text

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_punctuation_numbers(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_stopwords(text, stop_words):
    filtered_text = [word for word in text if word not in stop_words]
    return filtered_text

def stopwords_to_count(text, stop_words):
    n = len(text)
    j = 0
    while j < n:
        if text[j] not in stop_words:
            j += 1
            continue
        k = 0
        while j+k < n and text[j+k] in stop_words:
            k += 1
        text = text[:j] + [str(k)] + text[j+k:]
        n = n - k + 1
    return text


def stem_words(text):
    stems = [stemmer.stem(word) for word in text]
    return stems

def lemma_words(text):
    lemmas = [lemmatizer.lemmatize(word) for word in text]
    return lemmas

def preprocessing(text):
    stop_words = set(stopwords.words("english"))
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    
    for i in range(len(sentences)):
        sentences[i] = remove_uppercase(sentences[i])
        sentences[i] = remove_punctuation_numbers(sentences[i])
        sentences[i] = word_tokenize(sentences[i])
        sentences[i] = stopwords_to_count(sentences[i], stop_words)
        sentences[i] = lemma_words(sentences[i])
    
    return sentences


def co_occurance(sentences, windowSize):
    vocab = set()
    d = defaultdict(int)

    for sentence in sentences:
        for i in range(len(sentence)):
            if str.isdigit(sentence[i]): continue

            vocab.add(sentence[i])  
            for j in range(i, i+windowSize):
                if j >= len(sentence): break
                if str.isdigit(sentence[j]):
                    j += int(sentence[j])
                    continue
                key = tuple( sorted([sentence[j], sentence[i]]) )
                d[key] += 1
                
    vocab = sorted(vocab) # sort vocab

    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value

    return df


# ------------------------------------------------------------------------------------------------------------



