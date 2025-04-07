from pyvis.network import Network
import pandas as pd
import numpy as np

def word_network():
    df = pd.read_csv("co_occurrence_matrix.csv", index_col=0)
    net = Network()

    threshold = np.percentile(df.values, 95)

    nodes = list(df.columns)
    net.add_nodes(nodes)
    for source in df.keys():
        for destination in df.keys():
            if df.at[source, destination] > threshold:
                net.add_edge(source, destination, value=float(df.at[source, destination]))

    print(len(net.edges))

    net.force_atlas_2based(gravity=-50, central_gravity=0.005, spring_length=100, spring_strength=0.01)
  
    net.write_html("word_network.html")

#word_network()