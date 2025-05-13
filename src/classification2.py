from sentence_transformers import SentenceTransformer
from gensim.models.keyedvectors import KeyedVectors
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib as plt
import pandas as pd
import numpy as np
import random
import os

# Load words from file
def ld_words(filename):
    with open(filename, "r", encoding='utf-8') as f:    
        return f.read().splitlines()

# Load filenames from folder, all or n random
def get_file_names(path, n=None):
    file_names = os.listdir(path)
    if n != None:
        file_names = random.sample(file_names, n)
    return file_names

# Make dataframe of categories and words from files
def mk_words_df(path, files, include_category=False):
    data = {}

    for f in files:
        words = ld_words(f"{path}/{f}")
        category = f.split("-")[0]
        if include_category:
            words.append(category)
        data[category] = words

    df = pd.DataFrame.from_dict(data, orient='index').transpose()
    return df

def rm_all_duplicates(df):
    all_words = df.values.flatten()
    word_counts = pd.Series(all_words).value_counts()

    duplicates = set(word_counts[word_counts > 1].index)
    mask = df.apply(lambda col: ~col.isin(duplicates))

    return df.where(mask)


def make(path, n_files=None):
    files = get_file_names(path, n_files)
    msg = ""

    if not n_files:
       msg += f"All files from '{path}'\n"
    else:
        msg += f"Files from {path}:\n"
        for f in files: msg += f"{f}\n"
    msg += "\n"

    df = mk_words_df(path, files)
    n_before = df.count().sum()
    df = rm_all_duplicates(df)
    n_after = df.count().sum()

    msg += (f"Words:\n"+
            f"Total Before: {n_before}\n"+
            f"Removed: {n_before-n_after}, {(n_before-n_after)/n_before*100:.2f}%\n"+
            f"Total After: {n_after}"
           )
    return df, msg

def save_word_category_df(path, df):
    df_melted = df.melt(var_name="category", value_name="word").dropna()
    os.makedirs(path, exist_ok=True)
    df_melted[["word", "category"]].to_csv(f"{path}/categories_words.csv", index=False)

def ld_category_words_df(path):
    return pd.read_csv(f"{path}/categories_words.csv")

def ld_words_df(path):
    df = pd.read_csv(f"{path}/categories_words.csv")
    return df["word"].dropna().tolist()

def get_embeddings(words, model):
    if isinstance(model, KeyedVectors): 
        return np.array([model[word] for word in words if word in model])
    if isinstance(model, SentenceTransformer): 
        return model.encode(words, convert_to_tensor=False)
    raise TypeError("Unsupported model type")

def mk_palette(unique_labels, true_Labels):
    cmap = plt.colormaps.get_cmap("tab10")     
    label_to_color = {
        label: cmap(i % 10) 
        for i, label in enumerate(unique_labels)
    }
    true_colors = [
        label_to_color[label] 
        for label in true_Labels
    ]  
    return true_colors

def mk_kmeans_predictions(vectors, n):
    kmeans = KMeans(n_clusters=n, random_state=42)
    return kmeans.fit_predict(vectors)

def count_corrects(unique_labels, true_labels, predict_labels):
    label_to_int = { label: idx 
         for idx, label in enumerate(unique_labels)
    }
    true_int_labels = [ label_to_int[l] 
        for l in true_labels
    ]
    conf_mx = confusion_matrix(true_int_labels, predict_labels)
    row_ind, col_ind = linear_sum_assignment(-conf_mx)
    cluster_to_label = { cluster: label 
        for label, cluster in zip(row_ind, col_ind)
    }

    true_int_labels = np.array([label_to_int[c] for c in true_labels])
    mapped_preds = np.array([cluster_to_label[p] for p in predict_labels])

    corrects = Counter()
    wrongs = []
    for true_label, pred_label in zip(true_int_labels, mapped_preds):
        if true_label == pred_label:
            corrects[true_label] += 1

    corrects_named = {
        unique_labels[label]: count for label, count in corrects.items()
    }
    return corrects_named

   
models = {"MPNet-No-Tuning": SentenceTransformer("all-mpnet-base-v2"),

         }
 
save_path = "data/classification"
test_type = "similar"
n_files = 4
n_tests = 1

for t in range(n_tests):
    match test_type:
        case "distinct":
            data_path = "data/categories/all"
            test_name = f"distinct_test{t}"

        case "similar":
            data_path = "data/categories/similar"
            test_name = f"similar_test{t}"
            folders = os.listdir(data_path)
            data_path = f"{data_path}/{random.sample(folders, 1)[0]}"

    # Prepare directory and log file
    dir_path = f"{save_path}/{test_name}"
    os.makedirs(dir_path, exist_ok=True)
    log = open(f"{dir_path}/log.txt", "w")   
    
    # Make the dataframe and save it
    df, msg = make(data_path, n_files)
    save_word_category_df(dir_path, df)

    # Load and prepare test data
    words = ld_words_df(dir_path)
    df = ld_category_words_df(dir_path)
    word_category_dic = dict(zip(df['word'], df['category']))
    true_labels = [i for i in word_category_dic.values()]
    unique_labels = sorted(set(true_labels))   

    # List of colors alligned with true_labels
    colors = mk_palette(unique_labels, true_labels)            

    for model_name, model in models.items():
        embeddings = get_embeddings(words, model)
        predict_labels = mk_kmeans_predictions(embeddings, len(unique_labels))
        corrects = count_corrects(unique_labels, true_labels, predict_labels)

    print(corrects)
    log.write(msg)




