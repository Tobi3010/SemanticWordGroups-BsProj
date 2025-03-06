from pyvis.network import Network
import pandas as pd

def word_network():
    df = pd.read_csv('co_occurrence_matrix.csv')
    net = Network()
    print(df.columns[1:])
    net.add_nodes(df.columns[1:])
    return net

