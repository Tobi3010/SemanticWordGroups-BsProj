from sentence_transformers import SentenceTransformer
from gensim.models.keyedvectors import KeyedVectors
import gensim.downloader as gensim
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score,  mutual_info_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os

# Load words from file
def ld_words(filename):
    with open(filename, "r", encoding='utf-8') as f:    
        return f.read().splitlines()

# Load filenames from folder, all or n random
def get_file_names(path, n=None, log=None):
    file_names = os.listdir(path)
    if n != None:
        file_names = random.sample(file_names, n)

    if log != None:
        if n == None:
            log.write(f"All files from '{path}'")
        else:
            log.write(f"Files from {path}:\n")
            for f in file_names:
                log.write(f"{f}\n")

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


def make(path, files, log=None):
    
    df = mk_words_df(path, files)
    n_before = df.count().sum()
    df = rm_all_duplicates(df)
    n_after = df.count().sum()

    if log != None:

        log.write("\nWords:\n")
        log.write(f"Total Before: {n_before}\n")
        log.write(f"Removed: {n_before-n_after}, {(n_before-n_after)/n_before*100:.2f}%\n")
        log.write(f"Total After: {n_after}\n")
          
    return df

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
        vectors = []
        for word in words:
            if word in model:
                vectors.append(model[word])
            else: 
                print(word)
        return np.array(vectors)
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
    cluster_to_label = { cluster: label for cluster, label in zip(col_ind, row_ind) }

    true_int_labels = np.array([label_to_int[c] for c in true_labels])
    mapped_preds = np.array([cluster_to_label[p] for p in predict_labels])

    total = {label : 0 for label in set(true_int_labels)}
    corrects = {label : 0 for label in set(true_int_labels)}
    for true_label, pred_label in zip(true_int_labels, mapped_preds):
        total[true_label] += 1
        if true_label == pred_label:
            corrects[true_label] += 1

    corrects = { unique_labels[label]: count for label, count in corrects.items() }
    total = { unique_labels[label]: count for label, count in total.items() }

    return corrects, total


def mk_plot( words, embeddings, true_labels, unique_labels, colors):
    X_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=colors,
        s=60,
        marker='o',
    )

    for category in unique_labels:
        indices = [i for i, label in enumerate(true_labels) if label == category]
        centroid = np.mean(X_2d[indices], axis=0)
        ax.text(
            centroid[0], centroid[1], category,
            fontsize=11, weight='bold', ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
        )
    
    annotation = ax.annotate(
        "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "w"},
        arrowprops={"arrowstyle": "->"}
    )
    annotation.set_visible(False)

    def motion_hover(event):
        vis = annotation.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                index = ind["ind"][0]
                pos = scatter.get_offsets()[index]
                annotation.xy = pos
                word = words[index]
                category = true_labels[index]
                annotation.set_text(f"{word}\n({category})")
                annotation.get_bbox_patch().set_facecolor("lightyellow")
                annotation.set_visible(True)
                fig.canvas.draw_idle()
            elif vis:
                annotation.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", motion_hover)
    
    return plt


def main(models, n_tests, n_files, test_type, save_path):
    metrics = {}
    for model_name in models.keys():
        metrics[model_name] = {"TPC":0, "DB":0, "files":[]}

    for t in range(n_tests):
        match test_type:
            case "all":
                data_path = "data/categories/all"
                test_name = f"all_test{t}"
                folders = os.listdir(data_path)

            case "distinct":
                data_path = "data/categories/distinct"
                test_name = f"distinct_test{t}"

            case "similar":
                data_path = "data/categories/similar"
                test_name = f"similar_test{t}"
                folders = os.listdir(data_path)
                data_path = f"{data_path}/{random.sample(folders, 1)[0]}"
            
        # Prepare directory and log file
        dir_path = f"{save_path}/{test_name}"
        log_path = f"{dir_path}/log.txt"
        os.makedirs(dir_path, exist_ok=True)
        if os.path.exists(log_path): 
            os.remove(log_path)
        log = open(log_path, "a")   
        
        # Make the dataframe and save it
        files = get_file_names(data_path, n_files, log)
        df = make(data_path, files, log)
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
            corrects, total = count_corrects(unique_labels, true_labels, predict_labels)

            all_corrects = sum(corrects.values())
            all_total = sum(total.values())
            db_score = davies_bouldin_score(embeddings, predict_labels)
            ch_score = calinski_harabasz_score(embeddings, predict_labels)
            
            ari = adjusted_rand_score(true_labels, predict_labels)
            mi = mutual_info_score(true_labels, predict_labels)

            log.write("-" * 80 + "\n")
            log.write(f"{model_name} Clustering Results\n")
            log.write(f"{'Label':<20}{'Total':<20}{'Correct':<20}{'Percentage Correct':<20}\n")
            log.write("-" * 80 + "\n")

            for label in unique_labels:
                log.write(f"{label:<20}{total[label]:<20}{corrects[label]:<20}{corrects[label]/total[label]*100:<20.2f}\n")

            log.write("-" * 80 + "\n")
            log.write(f"Total Percentage Correct: {all_corrects/all_total*100:.2f}%\n")
            log.write(f"Davies-Bouldin Score: {db_score:.2f}\n")
            log.write(f"Calinski-Harabasz Index: {ch_score:.2f}\n")
            log.write(f"Adjusted Random Score: {ari:.2f}\n")
            log.write(f"Mutual Information Score: {mi:.2f}\n")
            log.write("-" * 80 + "\n")

            header = f"{model_name} Clustering Results\n"
            header += f"Total Accuracy: {all_corrects/all_total*100:.2f}%\n"
            for label in unique_labels:
                header += f"{label}: {corrects[label]/total[label]*100:.2f}%\n"
            
            plot = mk_plot(words, embeddings, true_labels, unique_labels, colors)
            plot.title(header)
            plot.tight_layout()
            plot.savefig(f"{dir_path}/{model_name}.png")

        log.close()
    



models = {
        "MPNet"              : SentenceTransformer("all-mpnet-base-v2"),
        "MEN-cossim-t-MPNet" : SentenceTransformer("data/models/MEN-cossim-t-MPNet"),
        "SEMCAT-Triple-t-MPNet"  : SentenceTransformer("data/models/SEMCAT-Triple-t-MPNet"),
        "MEN-cossim-SEMCAT-Triple-t-MPNet" : SentenceTransformer("data/models/MEN-cossim-SEMCAT-Triple-t-MPNet"),
        "SEMCAT-Triple-MEN-cossim-t-MPNet" : SentenceTransformer("data/models/SEMCAT-Triple-MEN-cossim-t-MPNet")
    }
 
save_path = "data/classification"
test_type = "similar"
n_files = 3 #max 4
n_tests = 5

main(models, n_tests, n_files, test_type, save_path)


