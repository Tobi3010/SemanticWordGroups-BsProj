from pyvis.network import Network

def word_network():
    net = Network()
    net.add_node(1, label="fire")
    net.add_node(2, label="water")
    return net