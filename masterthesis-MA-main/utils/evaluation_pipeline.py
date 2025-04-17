import pandas as pd
from py2neo import Graph
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector


class VectorSearchRefined:
    def __init__(
        self,
        url,
        password,
        username="neo4j",
        index_name="index_embedding",
        model_name="all-MiniLM-L6-v2",
        retrieval_query="",
    ):
        self.embedding_store = Neo4jVector.from_existing_index(
            HuggingFaceEmbeddings(model_name=model_name),
            url=url,
            username=username,
            password=password,
            index_name=index_name,
        )
        if retrieval_query:
            self.embedding_store.retrieval_query = retrieval_query
        else:
            self.embedding_store.retrieval_query = (
                f"RETURN node.title AS text, score, node.text AS add_info,"
                f"node {{.*, "
                f"`embedding`: Null, id: Null }} AS metadata"
            )
        self.graph_db = Graph(url, auth=(username, password))
        # Read relations for evaluation
        self.relations = pd.read_csv("./test_data.csv", sep=",")

    def perform_similarity_search(self, search_object, n_results):
        """ """
        self.search_object = search_object
        results = self.embedding_store.similarity_search_with_score(
            query=search_object["search_string"], k=n_results
        )
        page_list, text_list = [], []
        for i in range(len(results)):
            page_list.append(results[i][0].metadata["page_id"])
            if "text" in results[i][0].metadata:
                text_list.append(results[i][0].metadata["text"])
            else:
                text_list.append("")
        self.page_list = page_list
        self.text_list = text_list
        return page_list

    def execute_refinement(self, weight):
        """
        This function looks for all nodes in a range of 1-2 Edges
        """
        if self.page_list:
            query = f"""
                WITH {self.page_list} AS nodeIds
                MATCH (n:Page) 
                WHERE n.page_id IN nodeIds
                WITH collect(n) AS nodes
                UNWIND nodes AS n1
                UNWIND nodes AS n2
                WITH n1, n2 WHERE id(n1) < id(n2)
                MATCH p = allShortestPaths((n1)-[*1..2]-(n2))
                WHERE all(rel IN relationships(p) WHERE rel.weight > {weight})
                UNWIND nodes(p) AS n
                UNWIND relationships(p) AS r
                RETURN DISTINCT n.title AS title, n.page_id AS page_id, n.text AS text
            """
            self.refined = self.graph_db.run(query).data()
        else:
            raise ValueError(
                "No item id´s existing - Similarity Search must be executed prior to refinement!"
            )
        return self.refined

    def execute_refinement_maximized(self, weight):
        if self.page_list:
            query = f"""
                WITH {self.page_list} AS nodeIds
                MATCH (n:Page)
                WHERE n.page_id IN nodeIds
                WITH collect(n) AS nodes
                UNWIND nodes AS n1
                UNWIND nodes AS n2
                WITH n1, n2 WHERE id(n1) < id(n2)
                MATCH p = allShortestPaths((n1)-[*1..2]-(n2))
                WHERE all(rel IN relationships(p) WHERE rel.weight > {weight})
                UNWIND nodes(p) AS n
                UNWIND relationships(p) AS r
                RETURN DISTINCT n.title AS title, n.page_id AS page_id, n.text AS text
            """
            self.refined = self.graph_db.run(query).data()
        else:
            raise ValueError(
                "No item id´s existing - Similarity Search must be executed prior to refinement!"
            )
        return self.refined

    def return_refined_nodes(self):
        # Initialize an empty list to store the refined nodes
        refined_nodes_in_vs_result, self.refined_nodes_all = [], []

        # Iterate through each node in the self.refined list
        for node in self.refined:
            # Check if the current node's page_id is in the original page_list
            # This filters out nodes that were added during the refinement process
            self.refined_nodes_all.append(node["page_id"])
            if node["page_id"] in self.page_list:
                # If the node was in the original results, add it to refined_nodes
                refined_nodes_in_vs_result.append(node)

        # Store the filtered refined nodes as an instance variable
        self.refined_nodes = refined_nodes_in_vs_result

        # Return the list of refined nodes that were also in the original results
        return refined_nodes_in_vs_result

    def test_contained_items_with_score(self):
        filter_relevant_pages = self.relations[
            self.relations["query"] == self.search_object["query_id"]
        ]
        max_available_score = filter_relevant_pages["score"].sum()
        # print(max_available_score)
        relevant_pages = filter_relevant_pages["relevant_links"].to_list()
        scored_pages, scored_pages_refined = [], []
        for page in relevant_pages:
            if page in self.page_list:
                scored_pages.append(page)
            if page in self.refined_nodes_all:
                scored_pages_refined.append(page)

        score_vs = filter_relevant_pages[
            filter_relevant_pages["relevant_links"].isin(scored_pages)
        ]["score"].sum()
        score_vs_refined = filter_relevant_pages[
            filter_relevant_pages["relevant_links"].isin(scored_pages_refined)
        ]["score"].sum()

        return score_vs, score_vs_refined, max_available_score
