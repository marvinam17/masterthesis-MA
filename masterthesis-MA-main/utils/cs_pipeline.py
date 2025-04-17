import numpy as np
import pandas as pd
import networkit as nk
from scipy.sparse import coo_matrix
from tqdm import tqdm


def cs_refinement(Graph, res_cs, res, aimed_size: int):
    final_docs = []
    if len(res_cs) == aimed_size:
        return res_cs
    elif len(res_cs) > aimed_size:
        overlapping = [item for item in res if item in res_cs]
        final_docs.extend(overlapping)
        if len(overlapping) < aimed_size:
            high_degree_nodes = Graph.calculate_degree_of_nodes_in_subgraph(res_cs)
            for item in high_degree_nodes:
                if item not in overlapping:
                    final_docs.append(item)
                    if len(final_docs) == aimed_size:
                        return final_docs
            return final_docs
        else:
            return final_docs[:aimed_size]
    else:
        final_docs.extend(res_cs)
        for item in res:
            if item not in final_docs:
                final_docs.append(item)
                if len(final_docs) == aimed_size:
                    return final_docs
    raise ValueError


def apply_cs_component(ir_results, Graph, cs_method, core_node_count=10, **kwargs):
    """
    Apply the CS-Component to the results of the IR-Component
    """
    cs_res, cs_res_refined = {}, {}
    for query_id, res in tqdm(ir_results.items()):
        if query_id == "SemSearch_ES-3":
            continue
        # Get problem nodes
        problem_nodes = Graph.identify_problem_nodes(res)
        # Remove problem nodes from the result
        core_nodes = [node for node in res if node not in problem_nodes]
        # Get the top n core nodes
        if cs_method in ["LFMLocal", "GCE"]:
            core_nodes = Graph.ensure_two_nodes_with_degree_greater_0(
                core_nodes, core_node_count
            )
        else:
            core_nodes = core_nodes[:core_node_count]
        # Execute the community search
        cs_res[query_id] = Graph.execute_community_search(
            search_nodes=core_nodes, method=cs_method, **kwargs
        )

        # Add the problem nodes to the result
        cs_res[query_id].extend(problem_nodes)
        # Apply refinement to get the same number of nodes as in the IR-Component
        cs_res_refined[query_id] = cs_refinement(
            Graph, cs_res[query_id].copy(), res.copy(), len(res)
        )
    # Return results
    return cs_res_refined, cs_res


class GraphCS:
    def __init__(
        self,
        node_df: pd.DataFrame = pd.DataFrame(),
        rel_df: pd.DataFrame = pd.DataFrame(),
        load: bool = False,
        file_path: str = None,
    ):
        """
        Prefer Loading of the Graph in favour of Building it from scratch.
        Even though the creation from scratch might be faster.
        """
        self.VALID_METHODS = [
            "LocalTightnessExpansion",
            "LFMLocal",
            "LocalT",
            "GCE",
            "TwoPhaseL",
            "TCE",
            "PageRankNibble",
            "ApproximatePageRank",
        ]
        if load and file_path:
            print(
                """Warning: This is not recommended as it takes around 20 times longer than building from scratch. 
                Furthermore, a conversion from index to item_id is not possible without attribute data or node_df"""
            )
            self.G = nk.readGraph("./output/base_graph", nk.Format.NetworkitBinary)
            return
        elif load and not file_path:
            raise ValueError("No file path provided")
        elif not load and file_path:
            raise SyntaxError("Graph should not be loaded, but file path is provided")
        elif node_df.empty or rel_df.empty:
            raise ValueError("No node or relation data provided")
        else:
            self.node_df = node_df
            self.rel_df = rel_df
            self.item_to_index = node_df["id"].to_dict()
            self.inv_index_to_item = {v: k for k, v in self.item_to_index.items()}
            print("Building Graph from given Data...")
            self.G = self._build_graph_from_scratch()
            print("Graph built")
            return

    def _build_graph_from_scratch(self):
        """
        Builds a graph from the given node and relation data.
        """
        # Build Graph with size len(node_df)
        G = nk.Graph(len(self.node_df))
        # Prepare the relations
        # rel_df_existing_nodes_only = self._prepare_relations_for_graph()
        # Convert relations to numpy ndarrays and add them to the Graph
        G.addEdges(
            coo_matrix(
                (
                    np.ones(len(self.rel_df), dtype=np.int64),
                    (
                        self.rel_df["source"].to_numpy(),
                        self.rel_df["target"].to_numpy(),
                    ),
                )
            )
        )
        return G

    def ensure_two_nodes_with_degree_greater_0(self, core_nodes, core_node_count):
        init_nodes = []
        counter = 0
        for item in self._convert_dbpedia_to_index(core_nodes):
            degree = self.G.degree(item)
            if len(init_nodes) < core_node_count:
                init_nodes.append(item)
                if degree > 0:
                    counter += 1
            elif len(init_nodes) == core_node_count:
                if counter > 1:
                    break
                else:
                    if degree > 0:
                        counter += 1
                        if counter > 1:
                            init_nodes[-1] = item
                            break
                        else:
                            init_nodes[-2] = item
        return self._convert_index_to_dbpedia(init_nodes)

    def calculate_degree_of_nodes_in_subgraph(self, nodes):
        """
        Calculate the inner degree of each node.
        """
        nodes = self._convert_dbpedia_to_index(nodes)
        subG = nk.graphtools.subgraphFromNodes(self.G, nodes)
        degree_ranking = nk.centrality.DegreeCentrality(subG).run().ranking()
        return self._convert_index_to_dbpedia([node[0] for node in degree_ranking])

    def _execution_helper(self, cs, search_nodes):
        """
        Helperfunction to execute the community search.
        Parameters:
        cs: community search object
        """
        cs_result = cs.expandOneCommunity(search_nodes)
        self.cs_result = self._convert_index_to_dbpedia(cs_result)

    def _convert_index_to_dbpedia(self, node_list: list[int]):
        """
        Helperfunction to convert the index of the graph to dbpedia_id.
        Parameters:
        node_list: list of indices to convert
        """
        if self.item_to_index:
            return [self.item_to_index[node] for node in node_list]
        else:
            raise NotImplementedError("No index to item_id map available.")

    def _convert_dbpedia_to_index(self, node_list: list[str]):
        """
        Helperfunction to convert the dbpedia_id to the index.
        Parameters:
        node_list: list of indices to convert
        """
        if self.inv_index_to_item:
            return [self.inv_index_to_item[node] for node in node_list]
        else:
            raise NotImplementedError("No index to item_id map available.")

    def identify_problem_nodes(self, nodes: list, degree_limit: int = 100000):
        """
        Identify nodes with a degree higher than the limiter.
        Parameters:
        core_nodes: list of nodes to check
        degree_limit: degree limit
        """
        nodes = self._convert_dbpedia_to_index(nodes)
        return self._convert_index_to_dbpedia(
            [node for node in nodes if self.G.degree(node) > degree_limit]
        )

    def execute_community_search(
        self, search_nodes: list[int], method: str = "LocalTightnessExpansion", **kwargs
    ):
        """
        Execute the community search with the given method and parameters.
        Parameters:
        search_nodes: list of graph nodes to search for
        method: method to use for community search
        Returns:
        list of dbpedia_id in the community
        """
        search_nodes = self._convert_dbpedia_to_index(search_nodes)
        if method not in self.VALID_METHODS:
            raise ValueError("results: status must be one of %r." % self.VALID_METHODS)
        if method == "LocalTightnessExpansion":
            if kwargs.get("alpha"):
                self._execution_helper(
                    nk.scd.LocalTightnessExpansion(self.G, alpha=kwargs.get("alpha")),
                    search_nodes,
                )
            else:
                self._execution_helper(
                    nk.scd.LocalTightnessExpansion(self.G), search_nodes
                )
        elif method == "LFMLocal":
            if kwargs.get("alpha"):
                self._execution_helper(
                    nk.scd.LFMLocal(self.G, alpha=kwargs.get("alpha")), search_nodes
                )
            else:
                self._execution_helper(nk.scd.LFMLocal(self.G), search_nodes)
        elif method == "LocalT":
            self._execution_helper(nk.scd.LocalT(self.G), search_nodes)
        elif method == "GCE":
            if not kwargs.get("Q"):
                raise ValueError("No Q provided for GCE")
            else:
                self._execution_helper(
                    nk.scd.GCE(self.G, kwargs.get("Q")), search_nodes
                )
        elif method == "TwoPhaseL":
            self._execution_helper(nk.scd.TwoPhaseL(self.G), search_nodes)
        elif method == "ApproximatePageRank":
            if kwargs.get("alpha") and kwargs.get("epsilon"):
                nodes = nk.scd.ApproximatePageRank(
                    self.G, alpha=kwargs.get("alpha"), epsilon=kwargs.get("epsilon")
                ).run(search_nodes)
                return self._convert_index_to_dbpedia([node[0] for node in nodes])
            else:
                nodes = nk.scd.ApproximatePageRank(self.G, alpha=0.2, epsilon=0.01).run(
                    search_nodes
                )
                return self._convert_index_to_dbpedia([node[0] for node in nodes])
        elif method == "TCE":
            if kwargs.get("refine") and kwargs.get("useJaccard"):
                self._execution_helper(
                    nk.scd.TCE(
                        self.G,
                        refine=kwargs.get("refine"),
                        useJaccard=kwargs.get("useJaccard"),
                    ),
                    search_nodes,
                )
            else:
                self._execution_helper(nk.scd.TCE(self.G), search_nodes)
        elif method == "PageRankNibble":
            if kwargs.get("alpha") and kwargs.get("epsilon"):
                self._execution_helper(
                    nk.scd.PageRankNibble(
                        self.G, alpha=kwargs.get("alpha"), epsilon=kwargs.get("epsilon")
                    ),
                    search_nodes,
                )
            else:
                self._execution_helper(nk.scd.PageRankNibble(self.G))
        elif method == "ShortestPath":
            raise NotImplementedError("ShortestPath is not implemented yet")
        elif method == "RandomWalk":
            raise NotImplementedError("RandomWalk is not implemented yet")
        return self.cs_result
