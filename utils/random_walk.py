import random


def random_walk(G, start_node, steps):
    if start_node not in G:
        raise ValueError("Start node not in graph.")

    path = [start_node]
    current = start_node
    for _ in range(steps):
        neighbors = list(G.neighbors(current))
        if not neighbors:
            break
        current = random.choice(neighbors)
        path.append(current)
    return path
