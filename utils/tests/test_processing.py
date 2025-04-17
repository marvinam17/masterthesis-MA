from processing import evaluate_mrr
import pandas as pd


def test_evaluate_mrr():
    # Mock input data
    res_dict = {
        "vector": {"query1": [("doc1", 0.9), ("doc2", 0.8)]},
        "bm25": {"query1": [("doc2", 0.85), ("doc1", 0.75)]},
    }
    k = [1]

    # Mock evaluator class
    class MockEvaluator:
        def evaluate_custom(self, qrels, results, k_values, metric):
            return {"mrr@1": 0.5}

    evaluator = MockEvaluator()

    # Mock qrels
    eval_qrels = {"query1": {"doc1": 1, "doc2": 0}}

    # Run function
    result = evaluate_mrr(res_dict, k, evaluator, eval_qrels)

    # Assertions
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2  # One row per search type
    assert list(result.columns) == ["search_type", "mrr@1"]
    assert list(result["search_type"]) == ["vector", "bm25"]
