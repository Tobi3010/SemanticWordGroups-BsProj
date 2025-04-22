from data_handling import get_top_words, get_categories_words, save_words_df, load_words_df
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from pyvis.network import Network
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import community as community_louvain



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

def make_graph_from_df(df, top_percent):
    df_sorted = df.sort_values(by="cos_sim", ascending=False)
    top_n = int(len(df_sorted) * top_percent)
    df_top = df_sorted.head(top_n)

    G = nx.from_pandas_edgelist(df_top, 
                                source = "word1",
                                target = "word2",
                                edge_attr = "cos_sim",
                                create_using = nx.Graph())

    return G

def get_representative_nodes(G, communities):
    rep_nodes = {}
    degrees = dict(G.degree())

    for node, community in communities.items():
        if community not in rep_nodes:
            rep_nodes[community] = node
        else:
            if degrees[node] > degrees[rep_nodes[community]]:
                rep_nodes[community] = node

    return rep_nodes



categories = ["animal-268.txt", "colors-89.txt", "emotions-136.txt"]

words = get_categories_words(categories)
model = SentenceTransformer("data/models/MEN-t-MPNet")
df = make_nodes_df(words, model)
save_words_df(df)

df = load_words_df()
G = make_graph_from_df(df, 0.10)
pos = nx.kamada_kawai_layout(G)
nodes_degree = dict(G.degree)
communities = community_louvain.best_partition(G)

nx.set_node_attributes(G, nodes_degree, 'size')
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

# Optional: Fix nodes so they don't float on reload
net.write_html("word_network.html")



