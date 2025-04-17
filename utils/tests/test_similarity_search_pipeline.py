import os
import pytest
from similarity_search_pipeline import Search
import logging

API_KEY = os.getenv("ES_API_KEY")


@pytest.fixture
def init_search():
    client = Search(
        host="https://localhost:9200/",
        api_key=API_KEY,
        verify=False,
        index="test-index",
        vector_index="test_vector_index",
        embedding_model="sentence-transformers/multi-qa-mpnet-base-cos-v1",
    )
    yield client


def test_search(init_search, test_index, vector_index):
    """
    Test the similarity search functions.
    - functional correctness
    """

    res_bm25 = init_search.perform_similarity_search_bm25(
        search_string="test", n_results=3, return_objects=["dbpedia_id"]
    )
    res_boolean = init_search.perform_similarity_search_boolean(
        search_string="test", n_results=3, return_objects=["dbpedia_id"]
    )
    res_vector = init_search.perform_similarity_search_vector(
        search_string="test", n_results=3, return_objects=["dbpedia_id"]
    )
    # Should return 1, 2, 3 as dbpedia_id because test is in all documents
    assert (
        res_bm25["dbpedia_id"]
        == res_boolean["dbpedia_id"]
        == res_vector["dbpedia_id"]
        == ["1", "2", "3"]
    )


def test_boolean(init_search, caplog, test_index):
    """
    Test the boolean search functions.
    - construct validity
    - content validity
    - Stopword removal
    """
    res_boolean = init_search.perform_similarity_search_boolean(
        search_string="Wikipedia", n_results=1, return_objects=["dbpedia_id"]
    )
    # Should return 3 as dbpedia_id because Wikipedia is not in the other documents
    assert res_boolean["dbpedia_id"] == ["3"]
    # Should return 2 as score because Wikipedia is in both title and text
    assert res_boolean["score"] == [2.0]
    # reproducability:
    assert (
        res_boolean["dbpedia_id"]
        == init_search.perform_similarity_search_boolean(
            search_string="Wikipedia", n_results=1, return_objects=["dbpedia_id"]
        )["dbpedia_id"]
    )
    # Test stopword removal. Should return the same results as before because there are only stopwords added to the query.
    # Test logging as well
    with caplog.at_level(logging.INFO):
        res_boolean_stopword_removal = init_search.perform_similarity_search_boolean(
            search_string="Wikipedia is a of a",
            n_results=3,
            return_objects=["dbpedia_id"],
        )
    assert "Less results found than expected" in caplog.text
    assert res_boolean_stopword_removal["score"] == [2.0]
    assert res_boolean_stopword_removal["dbpedia_id"] == ["3"]


def test_bm25(init_search, test_index):
    """
    Test the boolean search functions.
    - construct validity
    - content validity
    """
    res_bm25 = init_search.perform_similarity_search_bm25(
        search_string="Wikipedia", n_results=1, return_objects=["dbpedia_id"]
    )
    # Should return 3 as dbpedia_id because Wikipedia is not in the other documents
    assert res_bm25["dbpedia_id"] == ["3"]
    # Should return 2 as score because Wikipedia is in both title and text
    assert res_bm25["score"] == [1.5082648]
    # reproducability:
    assert (
        res_bm25["dbpedia_id"]
        == init_search.perform_similarity_search_bm25(
            search_string="Wikipedia", n_results=1, return_objects=["dbpedia_id"]
        )["dbpedia_id"]
    )


def test_vector(init_search, vector_index):
    """
    Test the boolean search functions.
    - construct validity
    - content validity
    """
    res_vector = init_search.perform_similarity_search_vector(
        search_string="Wikipedia", n_results=1, return_objects=["dbpedia_id"]
    )
    # Should return 3 as dbpedia_id because Wikipedia is not in the other documents
    assert res_vector["dbpedia_id"] == ["3"]
    # reproducability:
    assert (
        res_vector["dbpedia_id"]
        == init_search.perform_similarity_search_vector(
            search_string="Wikipedia", n_results=1, return_objects=["dbpedia_id"]
        )["dbpedia_id"]
    )
