from pyvis.network import Network
import pandas as pd


def word_network():
    df = pd.read_csv("co_occurrence_matrix.csv", index_col=0)
    net = Network()

    nodes = list(df.columns)
    net.add_nodes(nodes)
    for source in df.keys():
        for destination in df.keys():
            if df.at[source, destination] > 0:
                net.add_edge(source, destination, value = int(df.at[source, destination]))
    net.write_html("word_network.html")



