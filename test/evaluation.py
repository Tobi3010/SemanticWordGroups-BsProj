import gensim.downloader as api
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from data_handling import load_standard

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


def test_neural_model(model, standard, top_k=10):
    wordsim_scr_lst = []  # Human similarity scores
    cos_sim_lst = []      # Model cosine similarities
    error_list = []       # Track absolute errors for word pairs

    for (wrd1, wrd2), human_score in standard.items():
        # Get embeddings
        emb1 = model.encode(wrd1, convert_to_tensor=True)
        emb2 = model.encode(wrd2, convert_to_tensor=True)
        cos_sim = util.cos_sim(emb1, emb2).item()
        
        # Store scores
        wordsim_scr_lst.append(human_score)
        cos_sim_lst.append(cos_sim)

        # Track error
        error = abs(cos_sim - human_score)
        error_list.append(((wrd1, wrd2), human_score, cos_sim, error))

    # Compute Spearman correlation
    spearman, _ = spearmanr(cos_sim_lst, wordsim_scr_lst)

    # Sort by highest error (worst predictions)
    error_list.sort(key=lambda x: x[3], reverse=True)
    hardest_pairs = error_list[:top_k]

    print(f"Spearman Correlation: {spearman:.4f}\n")
    print(f"Top {top_k} hardest word pairs (largest discrepancy):")
    for (w1, w2), human, model_sim, err in hardest_pairs:
        print(f"  ({w1}, {w2}) - Human: {human:.2f}, Model: {model_sim:.2f}, Error: {err:.2f}")

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


# EVALUATION STANDARDS -----------------------------------------------------------------------------------------




# Evaluates Stastical Models -----------------------------------------------------------------------------------
def eval_stastistical():
    # Load list of statistical models
    static_models = {
        "Word2Vec"  : api.load("word2vec-google-news-300"),
        "GloVe"     : api.load("glove-wiki-gigaword-300"),
        "FastText"  : api.load("fasttext-wiki-news-subwords-300")
    }
    print("\nEVALUATE statistical MODELS")
    print("\tStatistical Models on Relatedness standards")
    for standard_name, standard in relatedness_standards.items():
        print(f"\t\tEvaluating {standard_name}")
        for model_name, model in static_models.items():   
            spearman = test_static_model(model, standard)
            print(f"\t\t\t{model_name} Spearman Correlation: {spearman:.4f}")
            
    print("\n\tStatistical Models on Similarity Standards")
    for standard_name, standard in similarity_standards.items():
        print(f"\t\tEvaluating {standard_name}")
        for model_name, model in static_models.items():   
            spearman = test_static_model(model, standard)
            print(f"\t\t\t{model_name} Spearman Correlation: {spearman:.4f}")


# Evaluates Neural Models -------------------------------------------------------------------------------------
def eval_neural():
    # Load list of neural models
    neural_models = {
        "MPNet"      : SentenceTransformer("all-mpnet-base-v2"),
        "T5"         : SentenceTransformer("sentence-transformers/sentence-t5-large"),
        "Roberta"    : SentenceTransformer("all-roberta-large-v1")
    }
    print("\nEVALUATE NEURAL MODELS")
    print("\tNeural Models on Relatedness standards")
    for standard_name, standard in relatedness_standards.items():
        print(f"\t\tEvaluating {standard_name}")
        for model_name, model in neural_models.items():   
            spearman = test_neural_model(model, standard)
            print(f"\t\t\t{model_name} Spearman Correlation: {spearman:.4f}")

    print("\n\tNeural Models on Similarity standards")
    for standard_name, standard in similarity_standards.items():
        print(f"\t\tEvaluating {standard_name}")
        for model_name, model in neural_models.items():   
            spearman = test_neural_model(model, standard)
            print(f"\t\t\t{model_name} Spearman Correlation: {spearman:.4f}")

# Evaluates Fine Tuned Models --------------------------------------------------------------------------------
def eval_fine_tuned(modesl):
    

    print("\nEVALUATE FINE TUNED MODELS")
    print("\tFine Tuned Models on Relatedness standards")
    for standard_name, standard in relatedness_standards.items():
        print(f"\t\tEvaluating {standard_name}")
        for model_name, model in models.items():   
            spearman = test_neural_model(model, standard)
            print(f"\t\t\t{model_name} Spearman Correlation: {spearman:.4f}")

    print("\n\tFine Tuned Models on Similarity standards")
    for standard_name, standard in similarity_standards.items():
        print(f"\t\tEvaluating {standard_name}")
        for model_name, model in models.items():   
            spearman = test_neural_model(model, standard)
            print(f"\t\t\t{model_name} Spearman Correlation: {spearman:.4f}")


#eval_stastistical()
#eval_neural()

# Load list of relatedness golden standards
relatedness_standards = {
    #"Wordsim-353-Relatedness"   : load_standard("data/relatedness/WordSim353-REL/full.txt"),
    #"EN-MTurk-287"              : load_standard("data/relatedness/EN-MTurk-287/full.txt"),
    #"EN-MTurk-771"              : load_standard("data/relatedness/EN-MTurk-771/full.txt"),
    #"MEN"                       : load_standard("data/relatedness/MEN/test.txt")
}
# Load list of similarity golden standards
similarity_standards = {
    #"Wordsim-363-Similarity"    : load_standard("data/similarity/WordSim353-SIM/full.txt"),
    #"SimLex-999"                : load_standard("data/similarity/SimLex-999/full.txt"),
    "SimVerb-3500(20%)"         : load_standard("data/similarity/SimVerb-3500/test.txt")
}

models = {"T5"      : SentenceTransformer("sentence-transformers/sentence-t5-large"),
}

eval_fine_tuned(models)









"""
# Essemble method -------------------------------------------------------------------------------------
print("\nEVALUATE ESSEMBLE METHOD")

for standard_name, standard in standards.items():
    print(f"\t Evaluating {standard_name}")
    spearman = test_essemble_method(neural_models["BERT"], neural_models["T5"], standard)
    print(f"\t\tEssemble method, using BERT and FastTet, Spearman Correlation: {spearman:.4f}")
"""





# T5 Model ------------------------------------------------------------------------------------------
"""
print("Testing T5 Models")
T5_model_name = "google/flan-t5-large"
T5_tokenizer = AutoTokenizer.from_pretrained(T5_model_name)
T5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_model_name)

print("Evaluating T5-style model on WordSim-353 (Relatedness)...")
spearman = test_T5_model(T5_model, T5_tokenizer, wordsim_relatedness)
print(f"T5-Style Spearman Correlation: {spearman:.4f}")

model = SentenceTransformer("all-roberta-large-v1")
for standard_name, standard in relatedness_standards.items():
    print(f"\tEvaluating {standard_name}")
    spearman = test_neural_model(model, standard)
    print(f"\t\tRoberta Spearman Correlation: {spearman:.4f}")

for standard_name, standard in similarity_standards.items():
    print(f"\tEvaluating {standard_name}")
    spearman = test_neural_model(model, standard)
    print(f"\t\tRoberta Spearman Correlation: {spearman:.4f}")
"""














