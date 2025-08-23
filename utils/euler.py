import networkx as nx

def has_eulerian_path(G):
    if G.is_directed():
        return nx.has_eulerian_path(G)
    else:
        return nx.has_eulerian_path(G)

def has_eulerian_cycle(G):
    if G.is_directed():
        return nx.is_eulerian(G)
    else:
        return nx.is_eulerian(G)
