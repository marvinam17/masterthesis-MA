import pandas as pd
from rerankers import Reranker
from reranking import (
    reciprocal_rank_fusion,
    reranking,
    apply_reranking_component,
    evaluate,
)


def test_evaluate():
    # Mock input data
    res_dict = {
        "vector": {
            "classic": {"query1": {"doc1": 0.9, "doc2": 0.8}},
            "reranked": {"query1": {"doc1": 0.95, "doc2": 0.85}},
            "reranked_cs": {"LTE": {"query1": {"doc1": 0.92, "doc2": 0.82}}},
        }
    }

    k = [1, 10]

    # Mock evaluator class
    class MockEvaluator:
        def evaluate(self, qrels, results, k_values):
            return [
                {
                    "precision@1": 0.9,
                    "recall@1": 0.4,
                    "precision@10": 0.8,
                    "recall@10": 0.7,
                }
            ]

    evaluator = MockEvaluator()

    # Mock qrels
    eval_qrels = {"query1": {"doc1": 1, "doc2": 0}}

    # Run evaluate function
    result_df = evaluate(res_dict, k, evaluator, eval_qrels)

    # Assertions
    assert len(result_df) == 3  # One row for each result strategy
    assert list(result_df.columns) == [
        "search_type",
        "result_strategy",
        "cs_algorithm",
        "precision@1",
        "recall@1",
        "precision@10",
        "recall@10",
    ]
    assert list(result_df["search_type"]) == ["vector", "vector", "vector"]
    assert list(result_df["result_strategy"]) == ["classic", "reranked", "reranked_cs"]
    assert list(result_df["cs_algorithm"]) == ["-", "-", "LTE"]


def test_apply_reranking_component():
    # Setup test data
    test_corpus = pd.DataFrame(
        {"id": ["doc1", "doc2"], "text": ["test document 1", "test document 2"]}
    )

    test_results = {"q1": ["doc1", "doc2"], "q2": ["doc2", "doc1"]}

    test_queries = {"q1": "test query 1", "q2": "test query 2"}

    test_ranker = Reranker(
        "mixedbread-ai/mxbai-rerank-base-v1", model_type="cross-encoder"
    )

    # Execute function
    result = apply_reranking_component(
        test_results, test_ranker, test_queries, test_corpus
    )

    # Assertions
    assert len(result) == 2
    assert "q1" in result
    assert "q2" in result
    assert len(result["q1"]) == 2
    assert len(result["q2"]) == 2
    assert isinstance(result["q1"], dict)
    assert isinstance(result["q2"], dict)
    assert all(isinstance(score, float) for score in result["q1"].values())
    assert all(isinstance(score, float) for score in result["q2"].values())


def test_reranking():
    # Create test data
    test_corpus = pd.DataFrame(
        {
            "id": ["doc1", "doc2", "doc3"],
            "text": ["test text 1", "test text 2", "test text 3"],
        }
    )

    # Mock ranker class
    class MockRanker:
        def rank(self, query, docs, doc_ids):
            class MockResult:
                def __init__(self, doc_id, score):
                    self.document = type("obj", (object,), {"doc_id": doc_id})
                    self.score = score

            class MockRankResult:
                def __init__(self):
                    self.results = [MockResult("doc1", 0.9), MockResult("doc2", 0.7)]

            return MockRankResult()

    mock_ranker = MockRanker()
    test_query = "test query title"
    test_docs = ["doc1", "doc2"]

    # Call function
    result = reranking(test_corpus, mock_ranker, test_query, test_docs)

    # Assert results
    assert isinstance(result, dict)
    assert len(result) == 2
    assert result["doc1"] == 0.9
    assert result["doc2"] == 0.7


def test_reciprocal_rank_fusion():
    # Test case 1: Basic functionality with two ranked lists
    ranked_lists = [["doc1", "doc2", "doc3"], ["doc2", "doc3", "doc1"]]
    result = reciprocal_rank_fusion(ranked_lists, k=60, n_results=100)

    # Check structure of result
    assert isinstance(result, dict)
    assert "dbpedia_id" in result
    assert "score" in result
    assert len(result["dbpedia_id"]) == len(result["score"])

    # Check if all docs are present
    assert set(result["dbpedia_id"]) == {"doc1", "doc2", "doc3"}

    # Check if scores are in descending order
    assert all(
        result["score"][i] >= result["score"][i + 1]
        for i in range(len(result["score"]) - 1)
    )

    # Test case 2: Empty lists
    assert reciprocal_rank_fusion([],n_results=100) == {"dbpedia_id": [], "score": []}

    # Test case 3: Single list
    single_list = ["doc1", "doc2"]
    result = reciprocal_rank_fusion([single_list], k=60, n_results=100)
    assert result["dbpedia_id"] == ["doc1", "doc2"]
    assert result["score"] == [1 / 61, 1 / 62]

    # Test case 4: Different k value
    result = reciprocal_rank_fusion([["doc1", "doc2"]], k=10, n_results=100)
    assert result["score"] == [1 / 11, 1 / 12]
