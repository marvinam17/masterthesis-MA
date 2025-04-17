from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkit as nk
import numpy as np


def calculate_community_density(graph, community_nodes):
    """
    Berechnet die Dichte einer Community in einem Graphen.

    Parameters:
        graph (nk.graph.Graph): Der Graph, in dem die Community definiert ist.
        community_nodes (list): Liste von Knoten, die zur Community gehören.

    Returns:
        float: Die Dichte der Community.
    """
    # Subgraph für die Community erstellen
    subgraph = nk.graphtools.subgraphAndNeighborsFromNodes(
        graph, community_nodes, includeOutNeighbors=False
    )

    # Anzahl der Kanten und Knoten in der Community
    num_edges = subgraph.numberOfEdges()
    num_nodes = subgraph.numberOfNodes()

    # Dichte berechnen: 2E / (n * (n - 1))
    if num_nodes <= 1:
        return 0  # Keine sinnvolle Dichte für leere oder einzelne Knoten

    density = (2 * num_edges) / (num_nodes * (num_nodes - 1))
    return density


def get_node_colors(res, qrels):
    colors = []
    for item in res:
        if item not in qrels.keys() or qrels[item] == 0:
            colors.append("lightgrey")
            continue
        if qrels[item] == 1:
            colors.append("darkgrey")
        elif qrels[item] == 2:
            colors.append("dimgrey")
    return colors


def get_relevant_labels(C, G, node_df, qrels):
    labels = {}
    for node in C.iterNodes():
        if G.item_to_index[node] in qrels.keys():
            if qrels[G.item_to_index[node]] > 0:
                labels[node] = node_df[node_df.index == node].title.values[0]
    return labels


def calculate_content_coherence(model_name, node_contents):
    """
    Berechnet die Kohärenz der Inhalte einer Community.

    Parameters:
        node_contents (list of str): Liste von Texten/Inhalten der Knoten in der Community.

    Returns:
        float: Durchschnittliche Kosinusähnlichkeit zwischen den Knoteninhalten (Kohärenz).
    """
    # SentenceTransformer-Modell laden
    model = SentenceTransformer(model_name)

    # Embeddings für die Knoteninhalte berechnen
    embeddings = model.encode(node_contents, convert_to_tensor=False)

    # Kosinusähnlichkeit zwischen allen Knotenpaaren berechnen
    similarity_matrix = cosine_similarity(embeddings)

    # Nur obere Dreiecksmatrix (ohne Diagonale) für Paarvergleiche verwenden
    n = len(node_contents)
    upper_triangle_indices = np.triu_indices(n, k=1)
    similarities = similarity_matrix[upper_triangle_indices]

    # Durchschnittliche Ähnlichkeit berechnen
    if len(similarities) == 0:
        return 0  # Keine Paare vorhanden

    average_similarity = np.mean(similarities)
    return average_similarity
