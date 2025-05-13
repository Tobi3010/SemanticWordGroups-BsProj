from data_handling import load_df_words, load_df_category_words
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Specify data and model to use herre
# distinct 
#data_name = "animal268-sports222-colors89"
#data_name = "art124-computer176"
#data_name = "office243-food277-pirate165-christmas125"
#data_name = "languages115-colors89"
data_name = "all"

# similar
#data_name = "emotions136-positive_words247-negative_words178"
#data_name = "grammar67-math100-computer176"
#data_name = "birds120-reptiles18-mythical_beasts50-insect36"
#data_name = "music_theory81-musical_instruments61-dance29"
#data_name = "winter84-spring45-summer62-fall37"

#model_name = "GloVe"
#model = api.load("glove-wiki-gigaword-300")

#model_name = "MPNet - No Tuning"
#model = SentenceTransformer("all-mpnet-base-v2")
#model_name = "MPNet - Cosine Similarity"
#model = SentenceTransformer("data/models/MEN-cossim-t-MPNet")

#model_name = "T5 - No Tuning"
#model = SentenceTransformer("sentence-transformers/sentence-t5-large")
model_name = "T5 - Relatived Relatedness Learning"
model = SentenceTransformer("data/models/SimVerb3500-rrl-t-T5")



# Actual Code ---------------------------------------------------

# Load and prepare data
words = load_df_words(data_name)    
df = load_df_category_words(data_name)                          # Load words and categories
word_to_category = dict(zip(df['word'], df['category']))        # Make dictionary, keys are words, values are category
true_categories = [word_to_category[w] for w in words]          # List of correct categories

# Load model and get word embeddings   
if model_name == "GloVe":
    embeddings = np.array([model[word] for word in words if word in model])
else:                        
    embeddings = model.encode(words, convert_to_tensor=False)

# Clustering 
unique_categories = sorted(set(true_categories))                # Sorted list of unique categories, must be sorted to ensure identical coloring through multiple runs
n_categories = len(unique_categories)                           # Number of categories

palette = sns.color_palette("tab10", n_colors=n_categories)     # Select colors from seaborn "tab10" palette
category_to_color = {}                                          # Make dictionary mapping from category to color     
for category, color in zip(unique_categories, palette):
    category_to_color[category] = color
true_colors = [category_to_color[c] for c in true_categories]   # List of true colors

kmeans = KMeans(n_clusters=n_categories, random_state=42)
predicted_labels = kmeans.fit_predict(embeddings)

category_to_int = {c: i for i, c in enumerate(unique_categories)}
true_int_labels = np.array([category_to_int[c] for c in true_categories])

conf_mat = confusion_matrix(true_int_labels, predicted_labels)
row_ind, col_ind = linear_sum_assignment(-conf_mat)
cluster_to_category = {cluster: category for category, cluster in zip(row_ind, col_ind)}

# Ensure all are numeric labels
true_int_labels = np.array([category_to_int[c] for c in true_categories])
mapped_preds = np.array([cluster_to_category[p] for p in predicted_labels])

# === Evaluation ===-------------------------------------------------------------------------------------------
misclassified_words = set()
for word, true_label, pred_label in zip(words, true_int_labels, mapped_preds):
    if true_label != pred_label:
        misclassified_words.add(word)
        #print(f"Word: {word:<15} | True: {unique_categories[true_label]:<10} | Predicted: {unique_categories[pred_label]}")

correct = sum(p == t for p, t in zip(mapped_preds, true_int_labels))
prc = correct/len(words)
ari = adjusted_rand_score(true_categories, predicted_labels)
nmi = normalized_mutual_info_score(true_categories, predicted_labels)



# Reduce to 2D with t-SNE 
X_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
fig, ax = plt.subplots(figsize=(8, 6))



# Plot with per-point edge colors
scatter = ax.scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=true_colors,
    s=60,
    marker='o',
)

# === Optional: Category labels at centroids ===
for category in unique_categories:
    indices = [i for i, label in enumerate(true_categories) if label == category]
    centroid = np.mean(X_2d[indices], axis=0)
    ax.text(
        centroid[0], centroid[1], category,
        fontsize=11, weight='bold', ha='center', va='center',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
    )

# annotation style
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
            category = true_categories[index]
            annotation.set_text(f"{word}\n({category})")
            annotation.get_bbox_patch().set_facecolor("lightyellow")
            annotation.set_visible(True)
            fig.canvas.draw_idle()
        elif vis:
            annotation.set_visible(False)
            fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", motion_hover)

# === Final touches ===
plt.title(
        f"{model_name} : \n"+ 
        f"Corretly Classified: {prc:.2f}%, {correct}\{len(words)}\n"
        f"Adjusted Random Score: {ari:.2f}\n"+
        f"Normalized Mutual Information: {nmi:.2f}")
plt.tight_layout()
plt.savefig('test.png')
plt.show()

