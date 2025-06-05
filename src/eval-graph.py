import community as louvain
import networkx as nx
import numpy as np
import pandas as pd



def load_words_df(path):
    return pd.read_csv(path)

def make_graph_from_df(df, top_percent):
    # Include only top perecent word pairs
    df_sorted = df.sort_values(by="cos_sim", ascending=False)
    top_n = int(len(df_sorted) * top_percent)
    df_top = df_sorted.head(top_n)

    G = nx.from_pandas_edgelist(
        df_top, 
        source = "word1",
        target = "word2",
        edge_attr = "cos_sim",
        create_using = nx.Graph()
    )

    return G

def graph_metrics(G):
    metrics = {}

    # Clustering coefficient
    metrics["clust_coef"] = nx.average_clustering(G, weight="cos_sim")

    # Average path length
    if nx.is_connected(G): # If everything forms one graph
        metrics["avg_path_l"] = nx.average_shortest_path_length(G, weight="cos_sim")
    else: # If network is multiple graphs, pick biggest subgraph
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        metrics["avg_path_l"] = nx.average_shortest_path_length(subgraph, weight="cos_sim")

    # Modularity: Measures strength of community structure
    partition = louvain.best_partition(G)
    metrics["modularity"] = louvain.modularity(partition, G, weight="cos_sim")
   
    # Edge weight variance: Are edge similarities well distributed?
    edge_weights = [e["cos_sim"] for w1, w1, e in G.edges(data=True)]
    metrics["edge_w_variance"] = np.var(edge_weights)

    return metrics


def main(metrics, graphs):
    # Convert old metrics to pandas frame
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={"index": "model"}, inplace=True)

    # Get metrics for all model graphs
    results = []
    for model_name, G in graphs.items():
        G_metrics = graph_metrics(G)
        result = {
            "model": model_name,
            **G_metrics,
        }
        print(result)
        results.append(result)

    G_metrics_df = pd.DataFrame(results)
    # Merge data frame so all metrics for each model are on the row
    merged_df = pd.merge(G_metrics_df , metrics_df, on="model")

    # Compute pearson correlations
    cor_matrix = merged_df.corr(numeric_only=True, method="pearson")
    # Only correlations between old metrics and graph metrics
    cor = cor_matrix.loc[ 
        ["clust_coef", "avg_path_l", "modularity", "edge_w_variance"], 
        ["similarity", "relatedness", "semcat_distinct", "semcat_similar"]
    ]

    print(cor.round(3))
   

# Models on previus metrics, see thesis paper
metrics = {
    # Statistical models
    "Word2Vec"      : {"similarity": 0.5175, "relatedness": 0.5685, "semcat_distinct": 0.7474, "semcat_similar": 0.6498},
    "GloVe"         : {"similarity": 0.4125, "relatedness": 0.5192, "semcat_distinct": 0.6982, "semcat_similar": 0.6694},
    "FastText"      : {"similarity": 0.5350, "relatedness": 0.5934, "semcat_distinct": 0.7251, "semcat_similar": 0.5723},
    # Contextual models
    "MPNet"         : {"similarity": 0.5887, "relatedness": 0.6041, "semcat_distinct": 0.7860, "semcat_similar": 0.7245},
    "T5"            : {"similarity": 0.6301, "relatedness": 0.5693, "semcat_distinct": 0.7147, "semcat_similar": 0.6743},
    "RoBERTa"       : {"similarity": 0.5150, "relatedness": 0.5461, "semcat_distinct": 0.7245, "semcat_similar": 0.6246},
    # Tuned moodels
    "MPNet-tuned"   : {"similarity": 0.5800, "relatedness": 0.7739, "semcat_distinct": 0.8288, "semcat_similar": 0.5957},
    "T5-tuned"      : {"similarity": 0.7854, "relatedness": 0.6982, "semcat_distinct": 0.7186, "semcat_similar": 0.6013}
}

G_path = "data/networks/science/science"
top = 0.15
graphs = {
    "Word2Vec"      : make_graph_from_df(load_words_df(G_path+"-Word2Vec.csv"), top),
    "GloVe"         : make_graph_from_df(load_words_df(G_path+"-GloVe.csv"), top),
    "FastText"      : make_graph_from_df(load_words_df(G_path+"-FastText.csv"), top),
    "RoBERTa"       : make_graph_from_df(load_words_df(G_path+"-RoBERTa.csv"), top),
    "MPNet"         : make_graph_from_df(load_words_df(G_path+"-MPNet.csv"), top),
    "T5"            : make_graph_from_df(load_words_df(G_path+"-T5.csv"), top),
    "MPNet-tuned"   : make_graph_from_df(load_words_df(G_path+"-MPNet-tuned.csv"), top),
    "T5-tuned"      : make_graph_from_df(load_words_df(G_path+"-T5-tuned.csv"), top)
}

pd.set_option("display.max_columns", None)
main(metrics, graphs)




