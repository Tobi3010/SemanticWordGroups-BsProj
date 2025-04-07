import gensim.downloader as api
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from datastorage2 import load_wordsim

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def test_static_model(model, standard):
    wordsim_scr_lst = [] # wordsim-353 score list
    cos_sim_lst = [] # cosine similarity score list
    for (wrd1, wrd2), scr in standard.items():
        if wrd1 in model and wrd2 in model:        
            # Get word embeddings
            emb1 = model[wrd1]                      
            emb2 = model[wrd2]                      
            cos_sim = 1 - cosine(emb1, emb2) # Cosine similarity
            # Append both scores, ensures equal order
            wordsim_scr_lst.append(scr)                 
            cos_sim_lst.append(cos_sim)   
    # Compute spearman correlation
    spearman, _ = spearmanr(cos_sim_lst, wordsim_scr_lst) 
    return spearman    


def test_neural_model(model, standard):
    wordsim_scr_lst = [] # wordsim-353 score list
    cos_sim_lst = [] # cosine similarity score list
    for (wrd1, wrd2), scr in standard.items():
        # Get word embeddings
        emb1 = model.encode(wrd1, convert_to_tensor=True)
        emb2 = model.encode(wrd2, convert_to_tensor=True)
        cos_sim = util.cos_sim(emb1, emb2).item()  # Cosine similarity
        # Append both scores, ensures equal order
        wordsim_scr_lst.append(scr)                 
        cos_sim_lst.append(cos_sim)

    spearman, _ = spearmanr(cos_sim_lst, wordsim_scr_lst)    
    return spearman

def test_essemble_method(model1, model2, standard):
    wordsim_scr_lst = [] # wordsim-353 score list
    cos_sim_lst = [] # cosine similarity score list
    for (wrd1, wrd2), scr in standard.items():
        # Get word embeddings
        m1_emb1 = model1.encode(wrd1, convert_to_tensor=True)
        m1_emb2 = model1.encode(wrd2, convert_to_tensor=True)
        m2_emb1 = model2.encode(wrd1, convert_to_tensor=True)
        m2_emb2 = model2.encode(wrd2, convert_to_tensor=True)
        cos_sim = util.cos_sim((m1_emb1 + m2_emb1)/2, (m1_emb2 + m2_emb2)/2).item()  # Cosine similarity
        # Append both scores, ensures equal order
        wordsim_scr_lst.append(scr)                 
        cos_sim_lst.append(cos_sim)

    spearman, _ = spearmanr(cos_sim_lst, wordsim_scr_lst)    
    return spearman


def t5_relatedness(wrd1, wrd2, tokenizer, model):
    prompt = f"What is the semantic similarity score between the words: \"{wrd1}\" and \"{wrd2}\"?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=5)
    score_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        return float(score_str)
    except ValueError:
        print(f"Could not parse score from: {score_str}")
        return 0.0 

def test_T5_model(model, tokenizer, standard):
    gold_scores = []
    predicted_scores = []

    for (w1, w2), score in standard.items():
        pred_score = t5_relatedness(w1, w2, tokenizer, model)
        gold_scores.append(score)
        predicted_scores.append(pred_score)

    spearman, _ = spearmanr(predicted_scores, gold_scores)
    return spearman




standards = {
    "Wordsim-353 Relatedness Gold Standard": load_wordsim("data/test/wordsim_relatedness_goldstandard.txt"),
    "Wordsim-363 Similarity Gold Standard": load_wordsim("data/test/wordsim_similarity_goldstandard.txt")
}

# Stastical Models -----------------------------------------------------------------------------------

print("\nEVALUATE STATIC MODELS")

static_models = {
    "Word2Vec": api.load("word2vec-google-news-300"),
    "GloVe": api.load("glove-wiki-gigaword-300"),
    "FastText": api.load("fasttext-wiki-news-subwords-300")
}

for standard_name, standard in standards.items():
    print(f"\tEvaluating {standard_name}")
    for model_name, model in static_models.items():   
        spearman = test_static_model(model, standard)
        print(f"\t\t{model_name} Spearman Correlation: {spearman:.4f}")


# Neural Models -------------------------------------------------------------------------------------

print("\nEVALUATE NEURAL MODELS")

neural_models = {
    "BERT": SentenceTransformer("all-mpnet-base-v2"),
    "T5": SentenceTransformer("sentence-transformers/sentence-t5-large")
}

for standard_name, standard in standards.items():
    print(f"\tEvaluating {standard_name}")
    for model_name, model in neural_models.items():   
        spearman = test_neural_model(model, standard)
        print(f"\t\t{model_name} Spearman Correlation: {spearman:.4f}")

# Essemble method -------------------------------------------------------------------------------------
print("\nEVALUATE ESSEMBLE METHOD")

for standard_name, standard in standards.items():
    print(f"\t Evaluating {standard_name}")
    spearman = test_essemble_method(neural_models["BERT"], neural_models["T5"], standard)
    print(f"\t\tEssemble method, using BERT and FastTet, Spearman Correlation: {spearman:.4f}")
    



# T5 Model ------------------------------------------------------------------------------------------
"""
print("Testing T5 Models")
T5_model_name = "google/flan-t5-large"
T5_tokenizer = AutoTokenizer.from_pretrained(T5_model_name)
T5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_model_name)

print("Evaluating T5-style model on WordSim-353 (Relatedness)...")
spearman = test_T5_model(T5_model, T5_tokenizer, wordsim_relatedness)
print(f"T5-Style Spearman Correlation: {spearman:.4f}")
"""














