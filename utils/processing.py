import json
import pandas as pd


def json_file_access(file_path: str, mode: str = "r", data: dict = None):
    """
    Liest oder schreibt ein JSON-File.
    """
    with open(file_path, mode) as file:
        if mode == "w" and data:
            file.write(json.dumps(data))
        else:
            return json.load(file)


def evaluate(res_dict: dict, k: list[int], evaluator, eval_qrels) -> pd.DataFrame:
    evaluation_results = []
    for search_type, value in res_dict.items():  # vector, bm25, boolean
        data = {
            "search_type": search_type,
        }
        evals = evaluator.evaluate(qrels=eval_qrels, results=value, k_values=k)
        for eval in evals:
            data.update(eval)
        evaluation_results.append(data)
    return pd.DataFrame(evaluation_results)


def evaluate_mrr(res_dict: dict, k: list[int], evaluator, eval_qrels) -> pd.DataFrame:
    evaluation_results = []
    for search_type, value in res_dict.items():  # vector, bm25, boolean
        data = {
            "search_type": search_type,
        }
        evals = evaluator.evaluate_custom(
            qrels=eval_qrels, results=value, k_values=k, metric="mrr"
        )
        for eval in [evals]:
            data.update(eval)
        evaluation_results.append(data)
    return pd.DataFrame(evaluation_results)
