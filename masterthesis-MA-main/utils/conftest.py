import os
import pytest
from elasticsearch import Elasticsearch, NotFoundError
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
import warnings

warnings.filterwarnings("ignore")

API_KEY = os.getenv("ES_API_KEY")


@pytest.fixture
def es_client():
    # Initialize the Elasticsearch client with the provided API key.
    client = Elasticsearch(
        "https://localhost:9200/", api_key=API_KEY, verify_certs=False
    )
    yield client
    client.close()


@pytest.fixture
def mock_data():
    return [
        {
            "dbpedia_id": "1",
            "text": "This is a test document about Tests",
            "title": "Test Document 1",
        },
        {
            "dbpedia_id": "2",
            "text": "This is a document about information retrieval",
            "title": "Information Retrieval",
        },
        {
            "dbpedia_id": "3",
            "text": "This is an abstract of a wikipedia page",
            "title": "Wikipedia Page",
        },
    ]


# Autouse fixture to create an index on startup and delete it after tests.
@pytest.fixture()
def test_index(es_client):
    index_name = "test-index"

    # Create the index if it does not exist.
    if not es_client.indices.exists(index=index_name):
        response = es_client.indices.create(index=index_name)
        print(f"Created index '{index_name}':", response)

    fields_mapping = {
        "properties": {
            "dbpedia_id": {"type": "text"},
            "text": {"type": "text"},
            "title": {"type": "text"},
            "text_boolean_v2": {"type": "text", "similarity": "boolean"},
            "title_boolean_v2": {"type": "text", "similarity": "boolean"},
        }
    }
    response = es_client.indices.put_mapping(index=index_name, body=fields_mapping)

    # Add some mock data into the index.
    # These documents can serve as test data for your retrieval experiments.
    mock_documents = [
        {
            "dbpedia_id": "1",
            "text": "This is a test document about Tests",
            "title": "Test Document 1",
            "id": "1",
            "text_boolean_v2": "This is a test document about Tests",
            "title_boolean_v2": "Test Document 1",
        },
        {
            "dbpedia_id": "2",
            "text": "This is a test document about information retrieval",
            "title": "Information Retrieval",
            "id": "2",
            "text_boolean_v2": "This is a test document about information retrieval",
            "title_boolean_v2": "Information Retrieval",
        },
        {
            "dbpedia_id": "3",
            "text": "This is a test abstract of a wikipedia page",
            "title": "Wikipedia Page",
            "id": "3",
            "text_boolean_v2": "This is a test abstract of a wikipedia page",
            "title_boolean_v2": "Wikipedia Page",
        },
    ]

    for doc in mock_documents:
        # The document ID is cast to a string, which is the typical format for Elasticsearch IDs.
        response = es_client.index(index=index_name, id=str(doc["id"]), body=doc)
        print(f"Inserted document with id {doc['id']}: {response}")

    es_client.indices.refresh(index=index_name)
    # Yield control to the tests.
    yield index_name

    # Delete the index after tests have completed.
    try:
        response = es_client.indices.delete(index=index_name)
        print(f"Deleted index '{index_name}':", response)
    except NotFoundError:
        print(f"Index '{index_name}' was not found during teardown.")


@pytest.fixture()
def vector_index(mock_data, es_client):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"
    )
    index_name = "test_vector_index"
    elastic_vector_search = ElasticsearchStore(
        es_url="https://localhost:9200",
        index_name=index_name,
        embedding=embeddings,
        es_api_key="QktuNGlKTUJyZmtvclV0N2tHSFA6Q0FRYmExLVhTemkySGpvalBFMndDUQ==",
        es_params={"verify_certs": False},
    )
    # Create Langchain Documents and concatenate title and text as the field to vectorize
    vector_ingest = []
    for row in mock_data:
        vector_ingest.append(
            Document(
                page_content=(
                    row["title"] + " " + row["text"]
                    if len(row["title"]) > 0
                    else row["text"]
                ),
                metadata={"dbpedia_id": str(row["dbpedia_id"])},
            )
        )
    ids = [row["dbpedia_id"] for row in mock_data]

    elastic_vector_search.add_documents(documents=vector_ingest, ids=ids)

    yield index_name

    try:
        response = es_client.indices.delete(index=index_name)
        print(f"Deleted index '{index_name}':", response)
    except NotFoundError:
        print(f"Index '{index_name}' was not found during teardown.")
