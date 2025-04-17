import pandas as pd
from rerankers import Reranker
from tqdm import tqdm


def apply_reranking_component(results, ranker, queries, corpus):
    """
    Apply the Reranking-Component to the results of the prior Component
    """
    rerank_res = {}
    for query_id, res in tqdm(results.items()):
        try:
            rerank_res[query_id] = reranking(corpus, ranker, queries[query_id], res)
        except:
            print(query_id)
            continue
    return rerank_res


def reranking(
    corpus: pd.DataFrame, ranker: Reranker, query: str, to_be_reranked: list[str]
) -> dict[dict[str, float]]:
    """
    This function reranks the documents for a given query.
    Parameters:
    corpus: the complete corpus to get relevant text passages
    ranker: rerankers object
    query: the query to rerank for
    to_be_reranked: list of documents to rerank
    """
    reranked_scores = {}
    relevant_corpus = corpus[corpus["id"].isin(to_be_reranked)]
    reranked = ranker.rank(
        query=query,
        docs=relevant_corpus.text.to_list(),
        doc_ids=relevant_corpus.id.to_list(),
    )
    for doc in range(len(reranked.results)):
        reranked_scores[reranked.results[doc].document.doc_id] = reranked.results[
            doc
        ].score
    return reranked_scores


def convert_to_beir_evaluation(search_result: dict) -> dict:
    result_dict = {}
    for item in range(len(search_result["dbpedia_id"])):
        result_dict[search_result["dbpedia_id"][item]] = search_result["score"][item]
    return result_dict


def evaluate(res_dict: dict, k: list[int], evaluator, eval_qrels) -> pd.DataFrame:
    evaluation_results = []
    for search_type, value_1 in res_dict.items():  # vector, bm25, boolean
        for (
            result_strategy,
            value_2,
        ) in value_1.items():  # classic, reranked, reranked_cs
            if result_strategy == "reranked_cs":
                for cs_algorithm, value_3 in value_2.items():  # CS Methods
                    data = {
                        "search_type": search_type,
                        "result_strategy": result_strategy,
                        "cs_algorithm": cs_algorithm,
                    }
                    evals = evaluator.evaluate(
                        qrels=eval_qrels, results=value_3, k_values=k
                    )
                    for eval in evals:
                        data.update(eval)
                    evaluation_results.append(data)
            else:
                data = {
                    "search_type": search_type,
                    "result_strategy": result_strategy,
                    "cs_algorithm": "-",
                }
                evals = evaluator.evaluate(
                    qrels=eval_qrels, results=value_2, k_values=k
                )
                for eval in evals:
                    data.update(eval)
                evaluation_results.append(data)
    return pd.DataFrame(evaluation_results)


def perform_postprocessing_and_update_results(
    results,
    Graph,
    res,
    node_df,
    cs_algorithms: list[str],
    limiter,
    query: str,
    query_key,
    ranker,
    **kwargs,
):
    """
    This function is called after the retrieval phase of the pipeline.
    It is used to perform postprocessing on the results of the retrieval phase.

    res_dict = {
    "vector":{
        "classic": sim_search_results,
        "reranked": reranking_results,
        "reranked_cs": {"LTE": reranking_results_cs}
    }
    }
    """
    results["classic"].update({query_key: convert_to_beir_evaluation(res)})
    results["reranked"].update(
        {query_key: reranking(node_df, ranker, query, res["dbpedia_id"])}
    )
    # Check the Node Degrees
    cs_nodes = res["graph_id"][:limiter]
    removed_nodes = []
    for i, node in enumerate(cs_nodes):
        if Graph.G.degree(node) > 100000:
            removed_node = cs_nodes.pop(i)
            removed_nodes.append(removed_node)

    for cs in cs_algorithms:
        if cs not in results["reranked_cs"]:
            results["reranked_cs"][cs] = {}
        res_cs = Graph.execute_community_search(
            search_nodes=cs_nodes, method=cs, **kwargs
        )
        res_cs.extend(removed_nodes)
        res_cs = cs_refinement(res_cs, res, 100)
        results["reranked_cs"][cs].update(
            {query_key: reranking(node_df, ranker, query, res_cs)}
        )
    return results


def cs_refinement(res_cs, res, aimed_size: int):
    if len(res_cs) == aimed_size:
        # print("No Refinement needed")
        return res_cs
    elif len(res_cs) > aimed_size:
        # print("Refining too large community")
        # Prefer Overlapping Nodes from the Classic Search
        overlapping = [item for item in res if item in res_cs]
        if len(overlapping) < aimed_size:
            # print("Had less overlapping results. Have to add some from the original results")
            for item in res:
                if item not in overlapping:
                    overlapping.append(item)
                    if len(overlapping) == aimed_size:
                        return overlapping
        else:
            return res_cs[:aimed_size]
    else:
        # print("Refining too small community")
        for item in res:
            if item not in res_cs:
                res_cs.append(item)
                if len(res_cs) == aimed_size:
                    return res_cs
    print(
        f"Refinement failed - Less nodes in ir results - Length of res_cs: {len(res_cs)}"
    )
    return res_cs


def reciprocal_rank_fusion(ranked_lists, n_results, k=60):
    """
    Kombiniert Rankings mehrerer Retrieval-Methoden mittels Reciprocal Rank Fusion (RRF).

    Parameters:
        ranked_lists (list of lists): Liste von Rankings (eine Liste für jede Methode),
                                       wobei jede Liste die IDs oder Namen der Dokumente in Rangfolge enthält.
        n_results (int): Anzahl der Dokumente, die in das kombinierte Ranking aufgenommen werden sollen.
        k (int): Konstanter Parameter zur Steuerung der Gewichtung von Rangpositionen.

    Returns:
        dict: Kombinierte Scores als Dictionary Listen {dbpedia_id: [], score: []}, sortiert nach absteigenden Scores.
    """
    rrf_scores = {}
    target = {"dbpedia_id": [], "score": []}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            # Reciprocal Rank Fusion Formel: 1 / (k + rank)
            score = 1 / (k + rank)
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += score

    # Sortiere die Dokumente nach den RRF-Scores in absteigender Reihenfolge
    sorted_rrf_scores = dict(
        sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    )

    for i, (doc_id, score) in enumerate(sorted_rrf_scores.items()):
        if i < n_results:
            target["dbpedia_id"].append(doc_id)
            target["score"].append(score)

    return target
