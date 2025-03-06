import csv
import nltk
import string
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



# ------------------------------------------------------------------------------------------------------------

filename1 = "data/lyrics/tcc_ceds_music.csv"
filename2 = "data/lyrics/lyrics.csv"
filename3 = "data/lyrics/tcc_ceds_music.csv"
textdata = load_data(filename2)

# Example: Print the first lyric dictionary
if textdata:
    text = textdata[0]["lyrics"]
    text = remove_punctuation_numbers(text)
    text = remove_uppercase(text)
    text = remove_stopwords(text)
    text = lemma_words(text)
    
    print("\n Lyrics After Preprocessing ---------------------------------- \n")
    print(text) 
else:
    print("No data found.")

