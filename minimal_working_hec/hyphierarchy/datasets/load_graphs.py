import json

import networkx as nx


def load_graph_from_file(file_path: str) -> nx.DiGraph:
    with open(file_path, "r") as json_file:
        graph_dict = json.load(json_file)
    
    graph = nx.node_link_graph(graph_dict)

    return graph
