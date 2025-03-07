from preprocessing import *
from datastorage import *
from network import *

#sentence = "fire, but how, fires fire fire how fire, How fire but but fire how how but but but fire fire how water but"

filename2 = "data/lyrics/lyrics.csv"
stopwords = load_stopwords("data/stop_words_english.txt")

textdata = load_data(filename2)
text = textdata[0]["lyrics"]

sentences = preprocessing(text, stopwords)
df = co_occurance(sentences, 10)

df.to_csv("co_occurrence_matrix.csv", index=True)

