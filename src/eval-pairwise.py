
from sentence_transformers import SentenceTransformer
from gensim.models.keyedvectors import KeyedVectors
import gensim.downloader as gensim
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr


def load_standard(filename):
    dic = {}
    with open(filename, "r", encoding='utf-8') as f:   
        lines = f.read().splitlines()
        for i, line in enumerate(lines):
            parts = line.split("\t")
            if len(parts) != 3:
                print(f"Skipping malformed line {i + 1}: {line} ----------------------------------------")
                continue
            w1, w2, scr = parts
            key = tuple(sorted([w1, w2]))   
            dic[key] = float(scr)
    return dic


def test_model(model, standard):
    cossim_lst = []     # cosine similarity score list
    std_lst = []        # standard score list

    for (wrd1, wrd2), scr in standard.items():        
        # Get word embeddings
        if isinstance(model, KeyedVectors): # Statistical models
            if wrd1 in model and wrd2 in model:
                emb1 = model[wrd1]
                emb2 = model[wrd2]
        elif isinstance(model, SentenceTransformer): # Contextual models
            emb1 = model.encode(wrd1, convert_to_tensor=True)
            emb2 = model.encode(wrd2, convert_to_tensor=True)
                             
        cossim = 1 - cosine(emb1, emb2) 
        cossim_lst.append(cossim) 
        std_lst.append(scr)                 
          
    spearman, _ = spearmanr(cossim_lst, std_lst)  # Compute spearman correlation
    return spearman 


def evaluate(models, sets):
    # Load list of neural models

    for set_name, set in sets.items():
        print(f"{set_name} test")
        for standard_name, standard in set.items():
            print(f"\t{standard_name}:")
            for model_name, model, in models.items():
                spearman = test_model(model, standard)
                print(f"\t\t{model_name} Spearman : {spearman:.4f}")



# Load list of relatedness golden standards
relatedness_set = {
    "Wordsim-353-Relatedness"   : load_standard("data/relatedness/WordSim353-REL/full.txt"),
    "EN-MTurk-287"              : load_standard("data/relatedness/EN-MTurk-287/full.txt"),
    "EN-MTurk-771"              : load_standard("data/relatedness/EN-MTurk-771/full.txt"),
    "MEN"                       : load_standard("data/relatedness/MEN/test.txt")
}
# Load list of similarity golden standards
similarity_set = {
    "Wordsim-363-Similarity"    : load_standard("data/similarity/WordSim353-SIM/full.txt"),
    "SimLex-999"                : load_standard("data/similarity/SimLex-999/full.txt"),
    "SimVerb-3500(20%)"         : load_standard("data/similarity/SimVerb-3500/test.txt")
}

sets = {
    "Relatedness" : relatedness_set,
    "Similarity"  : similarity_set
}

models = {
    # Statistical modeles
    "Word2Vec"      : gensim.load("word2vec-google-news-300"),
    #"GloVe"         : gensim.load("glove-wiki-gigaword-300"),
    #"FastText"      : gensim.load("fasttext-wiki-news-subwords-300"),
    # Contextual models
    "MPNet"         : SentenceTransformer("all-mpnet-base-v2"),
    #"T5"            : SentenceTransformer("sentence-transformers/sentence-t5-large"),
    #"Roberta"       : SentenceTransformer("all-roberta-large-v1"),
    # Tuned models
    "MPNet-tuned"   : SentenceTransformer("data/models/MEN-cossim-t-MPNet"),
    #"T5-tuned"      : SentenceTransformer("data/models/SimVerb3500-rrl-t-T5")
}


evaluate(models, sets)