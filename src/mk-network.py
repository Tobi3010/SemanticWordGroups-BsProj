from sentence_transformers import SentenceTransformer
from gensim.models.keyedvectors import KeyedVectors
import gensim.downloader as gensim
from scipy.spatial.distance import cosine
from pyvis.network import Network
import community as louvain
import networkx as nx
import pandas as pd



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

            if isinstance(model, KeyedVectors): 
                emb1 = model[wrd1]
                emb2 = model[wrd2]
            elif isinstance(model, SentenceTransformer): 
                emb1 = model.encode(wrd1, convert_to_tensor=True)
                emb2 = model.encode(wrd2, convert_to_tensor=True)
            
            cos_sim = 1 - cosine(emb1, emb2)
            data.append((pair[0], pair[1], float(cos_sim)))

    df = pd.DataFrame(data, columns=["word1", "word2", "cos_sim"])
    return df

def mk_df_graph(df, top_percent):
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

def load_words(filename):
    with open(filename, "r", encoding='utf-8') as f:    
        return f.read().splitlines()

def get_categories_words(path, categories):
    stop_words = load_words("data/stop_words_english.txt")  # Stop words to ignore
    words = []

    for category in categories:
        words.extend(load_words(f"{path}/{category}"))  # Include all words from category
        words.append(category.split("-")[0])            # Include category name
    
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words


def mk_graph_df(categories, model, path):
    words = get_categories_words("data/categories/all", categories)
    df = make_nodes_df(words, model)
    df.to_csv(path, index=False)



def mk_network(path, top):
    df = pd.read_csv(path)
    G = mk_df_graph(df, top)
    pos = nx.kamada_kawai_layout(G)
    degree = dict(G.degree)
    communities = louvain.best_partition(G)

    nx.set_node_attributes(G, degree, "size")
    nx.set_node_attributes(G, communities, "group")

    net = Network(bgcolor="#222222", font_color="white")
    net.from_nx(G)

    # Scale edge weights 
    cos_sims = [edge[2]["cos_sim"] for edge in G.edges(data=True)]
    min_sim = min(cos_sims)
    max_sim = max(cos_sims)

    for edge in net.edges:
        sim = G[edge["from"]][edge["to"]]["cos_sim"]
        # Normalize similarity to scale 1 to 10
        width = 1 + 9 * ((sim - min_sim) / (max_sim - min_sim)) if max_sim > min_sim else 1
        edge["width"] = width

    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=150)

    # Get representative nodes
    representatives = get_representative_nodes(G, communities)

    # Assign group, color, and size to nodes
    for node in G.nodes(data=True):
        node_id = node[0]
        group = node[1].get("group", 0)
        
        # Default node properties
        net.get_node(node_id)["group"] = group
        net.get_node(node_id)["size"] = 15  # Default size

        if node_id in representatives.values():  # Highlight representatives
            net.get_node(node_id)["color"] = f"rgb({(group*50) % 255}, {(group*100) % 255}, {(group*150) % 255})"
            net.get_node(node_id)["size"] = 30  
            net.get_node(node_id)["borderWidth"] = 4  
            net.get_node(node_id)["borderColor"] = "black"  
        else:
            net.get_node(node_id)["color"] = f"rgb({(group*50) % 255}, {(group*100) % 255}, {(group*150) % 255})"  

    net.write_html("data/networks/word_network.html")


categories = ["science-70.txt"]
model = SentenceTransformer("data/models/MEN-cossim-t-MPNet")
path = "data/networks/word_similarity.csv"

mk_graph_df(categories, model, path)
mk_network(path, 0.10)