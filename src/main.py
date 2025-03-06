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

def remove_whitespace(text):
    return  " ".join(text.split())

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text

def stem_words(text):
    stems = [stemmer.stem(word) for word in text]
    return stems

def lemma_words(text):
    lemmas = [lemmatizer.lemmatize(word) for word in text]
    return lemmas





# Co-occurance ----------------------------------------------------------------------------------------------


def preprocessing(text):
    stop_words = set(stopwords.words("english"))
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    
    for i in range(len(sentences)):
        sentences[i] = sentences[i].lower()
        sentences[i] = re.sub(r'[^\w\s]', '', sentences[i])
        sentences[i] = word_tokenize(sentences[i])
        """
        j = 0
        while j < len(sentences[i]):
            if sentences[i][j] not in stop_words:
                j += 1
                continue

            k = 0
            while j < len(sentences[i]) and sentences[i][j] in stop_words:
                k += 1
                j += 1
                
            j = j - k
            sentences[i] = sentences[i][:j] + [str(k)] + sentences[i][j + k:]
        """
        sentences[i] = [word for word in sentences[i] if word not in stop_words]
        sentences[i] = [lemmatizer.lemmatize(word) for word in sentences[i]]
    
    return sentences


def co_occurance(sentences, windowSize):
    vocab = set()
    d = defaultdict(int)

    for sentence in sentences:
        for i in range(len(sentence)):
                token = sentence[i]
                vocab.add(token)  
                next_token = sentence[i + 1 : i + 1 + windowSize]
                for t in next_token:
                    key = tuple( sorted([t, token]) )
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

filename1 = "data/lyrics/tcc_ceds_music.csv"
filename2 = "data/lyrics/lyrics.csv"
filename3 = "data/lyrics/tcc_ceds_music.csv"
textdata = load_data(filename2)

# Example: Print the first lyric dictionary
if textdata:
    text = textdata[0]["lyrics"]
    sentences = preprocessing(text)
    df = co_occurance(sentences, 1)
    print(df)

    #text = remove_punctuation_numbers(text)
    #text = remove_uppercase(text)
    #text = remove_stopwords(text)
    #text = lemma_words(text)
    
else:
    print("No data found.")

