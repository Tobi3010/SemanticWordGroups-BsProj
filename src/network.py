from data_handling import *
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from pyvis.network import Network
import networkx as nx
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
import seaborn as sns


# Makes the nodes for 
def make_nodes_df(words, model):  
    data = []
    seen_pairs = set()
    
    for wrd1 in words:
        for wrd2 in words:
            if wrd1 == wrd2:
                continue
            pair = tuple(sorted([wrd1, wrd2]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            emb1 = model.encode(wrd1, convert_to_tensor=True)
            emb2 = model.encode(wrd2, convert_to_tensor=True)
            cos_sim = 1 - cosine(emb1, emb2)
            data.append((pair[0], pair[1], float(cos_sim)))

    df = pd.DataFrame(data, columns=["word1", "word2", "cos_sim"])
    return df


# Makes graph from top_percent of nodes
def make_graph_from_df(df, top_percent):
    df_sorted = df.sort_values(by="cos_sim", ascending=False)   # Sort in descending order, by cosine similarity
    top_n = int(len(df_sorted) * top_percent)                   # Index where to split for top words
    df_top = df_sorted.head(top_n)                              # Get the top words

    
    # Create graph from panda dataframe
    G = nx.from_pandas_edgelist(df_top, 
                                source = "word1",
                                target = "word2",
                                edge_attr = "cos_sim",
                                create_using = nx.Graph())
    
    return G

def make_network(G, communities, degree):
    nx.set_node_attributes(G, degree, 'size')
    nx.set_node_attributes(G, communities, 'group')
    net = Network(bgcolor='#222222', font_color='white')
    net.from_nx(G)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=150)

    # Get the representative nodes
    representatives = get_representative_nodes(G, communities)

    # Assign group, color, and size to nodes
    for node in G.nodes(data=True):
        node_id = node[0]
        group = node[1].get("group", 0)
        
        # Default node properties
        net.get_node(node_id)['group'] = group
        net.get_node(node_id)['size'] = 15  # Default size

        # Highlight the representative nodes
        if node_id in representatives.values():
            # Set the community color (same as group color)
            net.get_node(node_id)['color'] = f"rgb({(group*50) % 255}, {(group*100) % 255}, {(group*150) % 255})"
            net.get_node(node_id)['size'] = 30  # Larger size
            net.get_node(node_id)['borderWidth'] = 4  # Thicker border
            net.get_node(node_id)['borderColor'] = 'black'  # Black border for contrast
        else:
            net.get_node(node_id)['color'] = f"rgb({(group*50) % 255}, {(group*100) % 255}, {(group*150) % 255})"  # Same color as community
        
    return net


# Finds the representative node for each community
def get_representative_nodes(G, communities):
    rep_nodes = {}                   # Dictionary, keys are communities, values are representative node
    degrees = dict(G.degree())       # Node degree used for centrality

    # Loop through communities and their nodes
    # If another node has higher degree than current representative node
    # We make that node the new representative.
    for node, community in communities.items():
        if community not in rep_nodes:    # Start case, if community does not yet have representative  
            rep_nodes[community] = node
        else:                        
            rep_node = rep_nodes[community]
            if degrees[node] > degrees[rep_node]:  
                rep_nodes[community] = node

    return rep_nodes

def spectral_communities(G, n):
    A = nx.to_numpy_array(G)
    sc = SpectralClustering(n_clusters=n, affinity='precomputed', assign_labels='kmeans')
    labels = sc.fit_predict(A)
    return {node: int(label) for node, label in zip(G.nodes(), labels)}

def load_df_category_words(name):
    return pd.read_csv(f"data/networks/{name}/categories_words.csv")



print("Starting...")

categories = {"emotions-136.txt":0,"positive_words-247.txt":0, "negative_words-178.txt":0}
df = make_word_category_df(categories, False)
save_words_category_df(categories, df)
print("Word categories made and saved")
"""

parts = []
for category, n_words in categories.items():
    category_name = category.split("-")[0]
    parts.append(f"{category_name}{n_words}")
data_name = "-".join(parts)
"
data_name = "animal0-sports0-colors0"
model_name ="roberta"
#model = SentenceTransformer("all-roberta-large-v1")
#model_name ="MEN-cossim-t-MPNet"
#model = SentenceTransformer("data/models/MEN-cossim-t-MPNet")



words = load_df_words(data_name)
df = make_nodes_df(words, model)
save_df_cossim(df, data_name, model_name)
print("Word cosine similarity made and saved")

df = load_df_cossim(data_name, model_name)
G = make_graph_from_df(df, 0.15)
nodes_degree = dict(G.degree)
#communities = community_louvain.best_partition(G)
communities = spectral_communities(G, 2)
net = make_network(G, communities, nodes_degree)
net.write_html(f"data/networks/{data_name}/{model_name}-spec2.html")
print("Network made and saved")
"""




