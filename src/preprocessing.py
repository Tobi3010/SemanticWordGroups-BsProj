import string
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import words
import re

lemmatizer = WordNetLemmatizer()


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

def only_english(text):
    english_words = set(words.words())  # Convert list to set for faster lookup
    filtered_text = [word for word in text if word.lower() in english_words]
    return filtered_text

def lemma_words(text):
    lemmas = [lemmatizer.lemmatize(word) for word in text]
    return lemmas

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

def preprocessing(text, stop_words):
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    
    for i in range(len(sentences)):
        sentences[i] = remove_uppercase(sentences[i])
        sentences[i] = remove_punctuation_numbers(sentences[i])
        sentences[i] = word_tokenize(sentences[i])
        sentences[i] = stopwords_to_count(sentences[i], stop_words)
        sentences[i] = lemma_words(sentences[i])
    
    return sentences







