from preprocessing import *


#sentence = "fire, but how, fires fire fire how fire, How fire but but fire how how but but but fire fire how water but"

filename2 = "data/lyrics/lyrics.csv"
textdata = load_data(filename2)
text = textdata[0]["lyrics"]


sentences = preprocessing(text)
df = co_occurance(sentences, 10)
df.to_csv("co_occurrence_matrix.csv", index=True)