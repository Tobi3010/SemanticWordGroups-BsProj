import gensim.downloader as api
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from datastorage2 import load_wordsim




dic = load_wordsim("data/test/wordsim_relatedness_goldstandard.txt")
model = api.load("word2vec-google-news-300")

cosine_similarities = []
wordsim_scores = []
for key in dic.keys():
    word1, word2 = key[0], key[1]
    if word1 in model and word2 in model:
        embedding1 = model[word1]
        embedding2 = model[word2]
        similarity = 1 - cosine(embedding1, embedding2)
        
        cosine_similarities.append(similarity)
        wordsim_scores.append(dic[key])
    else:
        print(f"Skipping pair: ({word1}, {word2}) — not in model vocab")
        
corr, _ = spearmanr(cosine_similarities, wordsim_scores)
print(f"Spearman Correlation: {corr:.4f}")




