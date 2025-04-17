from cs_pipeline import GraphCS
import pandas as pd
import pytest


@pytest.fixture
def dummy_dataframes():
    """
    Create dummy nodes and relations DataFrames.

    The nodes DataFrame has a column 'id' with node identifiers.
    The relations DataFrame defines two communities:
      - Community 1: Nodes with id 0-4, fully connected.
      - Community 2: Nodes with id 5-9, fully connected.
    Two bridging edges are added between these communities.
    """
    # Nodes DataFrame with an explicit 'id' column.
    nodes_df = pd.DataFrame({"id": ["id_" + str(i) for i in range(10)]})

    edges = []
    # Community 1: Fully connected graph among nodes 0 to 4.
    for i in range(5):
        for j in range(i + 1, 5):
            edges.append((i, j))

    # Community 2: Fully connected graph among nodes 5 to 9.
    for i in range(5, 10):
        for j in range(i + 1, 10):
            edges.append((i, j))

    # Add bridging edges between the two communities.
    edges.append((2, 7))
    edges.append((4, 5))

    relations_df = pd.DataFrame(edges, columns=["source", "target"])

    return nodes_df, relations_df


@pytest.fixture
def dummy_graph(dummy_dataframes):
    """
    Create a GraphCS object from the dummy nodes and relations DataFrames.
    """
    nodes_df, relations_df = dummy_dataframes
    return GraphCS(nodes_df, relations_df)


def test_graph_creation(dummy_graph):
    """
    Test the creation of a GraphCS object and its basic properties.
    """

    # Check if the graph has the expected number of nodes and edges.
    assert dummy_graph.G.numberOfNodes() == 10
    assert dummy_graph.G.numberOfEdges() == 22


def test_community_search(dummy_graph):
    """
    Test the community search functionality of the GraphCS object.
    - construct validity
    - content validity
    - Tests conversion from index to dbpedia_id
    """
    # Perform community search LTE.
    res = dummy_graph.execute_community_search(
        ["id_0", "id_1"], "LocalTightnessExpansion", alpha=1.0
    )
    assert res == ["id_0", "id_1", "id_2", "id_3", "id_4"]
    res = dummy_graph.execute_community_search(
        ["id_6"], "LocalTightnessExpansion", alpha=1.0
    )
    assert res == ["id_5", "id_6", "id_7", "id_8", "id_9"]

    # Perform community search LFM.
    res = dummy_graph.execute_community_search(["id_0", "id_1"], "LFMLocal", alpha=1.0)
    assert res == ["id_0", "id_1", "id_2", "id_3", "id_4"]
    res = dummy_graph.execute_community_search(["id_6"], "LFMLocal", alpha=1.0)
    assert res == ["id_5", "id_6", "id_7", "id_8", "id_9"]

    # Perform community search GCE.
    res = dummy_graph.execute_community_search(["id_0", "id_1"], "GCE", Q="M")
    assert res == ["id_0", "id_1", "id_2", "id_3", "id_4"]
    res = dummy_graph.execute_community_search(["id_6"], "GCE", Q="M")
    assert res == ["id_5", "id_6", "id_7", "id_8", "id_9"]

    # Perform community search PRN.
    res = dummy_graph.execute_community_search(
        ["id_0", "id_1"], "PageRankNibble", alpha=0.1, epsilon=0.0001
    )
    assert res == ["id_0", "id_1", "id_2", "id_3", "id_4"]
    res = dummy_graph.execute_community_search(
        ["id_6"], "PageRankNibble", alpha=0.1, epsilon=0.0001
    )
    assert res == ["id_5", "id_6", "id_7", "id_8", "id_9"]

    # Perform community search TCE.
    res = dummy_graph.execute_community_search(["id_0", "id_1"], "TCE")
    assert res == ["id_0", "id_1", "id_2", "id_3", "id_4"]
    res = dummy_graph.execute_community_search(["id_6"], "TCE")
    assert res == ["id_5", "id_6", "id_7", "id_8", "id_9"]


def test_community_search_errors(dummy_graph):
    """
    Test error handling in the community search functionality.
    """
    # Test if error is thrown on wrong CS algorithm
    with pytest.raises(ValueError):
        dummy_graph.execute_community_search(["id_6"], "WrongAlgorithm", alpha=1.0)
    with pytest.raises(ValueError) as e:
        dummy_graph.execute_community_search(["id_6"], "GCE")
        # Test message
        assert "No Q provided for GCE" in str(e.value)


def test_problem_node_identification(dummy_graph, dummy_dataframes):
    """
    Test if problem nodes are correctly identified.
    id 2, 4, 5 and 7 have a higher degree than the threshold of 4 because they connect two communities.
    """
    nodes_df, _ = dummy_dataframes
    res = dummy_graph.identify_problem_nodes(nodes_df.id.to_list(), 4)
    assert res == ["id_2", "id_4", "id_5", "id_7"]


def test_calculate_degree_ranking_in_subgraph(dummy_graph):
    """
    Test the subgraph degree calculation.
    0-3 are in the same community and have a degree of 4.
    4 is a bridge node and has a degree of 5.
    5 is in the second community and has a degree of 1 because of the connection to 4
    """
    res = dummy_graph.calculate_degree_of_nodes_in_subgraph(
        ["id_0", "id_1", "id_2", "id_3", "id_4", "id_5"]
    )
    assert res == ["id_4", "id_0", "id_1", "id_2", "id_3", "id_5"]
