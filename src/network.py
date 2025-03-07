from pyvis.network import Network
import pandas as pd


def word_network():
    df = pd.read_csv("co_occurrence_matrix.csv")
    net = Network()

    nodes = list(df.columns)
    net.add_nodes(nodes[1:])
    return net

net = word_network()
net.write_html("word_network.html")


