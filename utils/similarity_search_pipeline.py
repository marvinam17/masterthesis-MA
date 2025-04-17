from langchain.embeddings import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

logger = logging.getLogger(__name__)


class Search:

    def __init__(self, host, api_key, verify, index, vector_index, embedding_model):
        self.client = Elasticsearch(host, api_key=api_key, verify_certs=verify)
        self.vector_store = ElasticsearchStore(
            es_url=host,
            index_name=vector_index,
            es_api_key=api_key,
            embedding=HuggingFaceEmbeddings(model_name=embedding_model),
            es_params={"verify_certs": verify},
        )
        self.index = index
        self.vector_index = vector_index

    def perform_similarity_search_vector(
        self, search_string, n_results, return_objects
    ):
        """
        Perform a similarity search based on the search_object and return a list of graph_ids.
        An Embedding Index is used to perform the similarity search.
        If there is the following error: BadRequestError: BadRequestError(400, 'illegal_argument_exception', '[num_candidates] cannot be less than [k]')
        then the number of candidates must be increased in elasticsearch.helpers.vectorstore._sync.vectorstore.VectorStore
        Currently that is not available in langchain integration. I made a suggestion to enable that:
        Reference: https://github.com/langchain-ai/langchain/issues/25180
        """
        results = self.vector_store.similarity_search_with_score(
            query=search_string,
            k=n_results,
        )
        deserialized = self._deserialize_results(results, return_objects)
        if len(deserialized[return_objects[0]]) <= n_results:
            logging.info("Less results found than expected")
        return deserialized

    def perform_similarity_search_bm25(self, search_string, n_results, return_objects):
        """
        Perform a similarity search based on the search_object and return a list of item IDs.
        bm25 is the default elastic search metric.
        """
        results = self.client.search(
            search_type="dfs_query_then_fetch",
            index=self.index,
            body=self._build_bm25_query(search_string, return_objects),
            size=n_results,
        ).body
        deserialized = self._deserialize_results(results, return_objects)
        if len(deserialized[return_objects[0]]) <= n_results:
            logging.info("Less results found than expected")
        return deserialized

    def perform_similarity_search_boolean(
        self, search_string, n_results, return_objects
    ):
        """
        Perform a similarity search based on the search_object and return a list of item IDs.
        """
        results = self.client.search(
            index=self.index,
            body=self._build_boolean_query(search_string, return_objects),
            size=n_results,
        ).body
        deserialized = self._deserialize_results(results, return_objects)
        if len(deserialized[return_objects[0]]) <= n_results:
            logging.info("Less results found than expected")
        return deserialized

    def perform_similarity_search(
        self, search_string, n_results, return_objects, search_type
    ):
        """
        Perform a similarity search based on the search_object and return a list of item IDs.
        """
        if search_type == "vector":
            return self.perform_similarity_search_vector(
                search_string, n_results, return_objects
            )
        elif search_type == "bm25":
            return self.perform_similarity_search_bm25(
                search_string, n_results, return_objects
            )
        elif search_type == "boolean":
            return self.perform_similarity_search_boolean(
                search_string, n_results, return_objects
            )
        else:
            raise ValueError(
                "Invalid search_type. Must be one of 'vector', 'bm25', or 'boolean'."
            )

    def _build_boolean_query(self, search_string, return_objects):
        """
        Build a boolean query based on the search_string.
        Had OR as default
        Removes english stopwords
        """
        search_string_tokens = word_tokenize(search_string)
        search_string_items = [
            w for w in search_string_tokens if w not in stopwords.words("english")
        ]
        must_query = [
            {"match": {"text_boolean_v2": item}} for item in search_string_items
        ]
        must_query.extend(
            [{"match": {"title_boolean_v2": item}} for item in search_string_items]
        )
        query = {
            "query": {"bool": {"should": must_query, "boost": 1.0}},
            "_source": return_objects,
        }
        return query

    def _build_bm25_query(self, search_string, return_objects):
        """
        Build a query based on the search_string.
        """
        query = {
            "query": {
                "multi_match": {
                    "query": search_string,
                    "type": "best_fields",
                    "fields": ["title", "text"],
                    "tie_breaker": 0.5,
                }
            },
            "_source": return_objects,
        }
        return query

    def _deserialize_results(self, results, return_objects):
        """
        Deserialize the results from the Elasticsearch response.
        """
        return_object = {}
        if isinstance(results, list):
            # BM25 and Boolean results
            for item in return_objects:
                item_list = []
                for i in range(len(results)):
                    item_list.append(results[i][0].metadata[item])
                return_object[item] = item_list
            return_object["score"] = [results[i][1] for i in range(len(results))]
        else:
            # VectorSearch results
            for item in return_objects:
                item_list = []
                for i in range(len(results["hits"]["hits"])):
                    item_list.append(results["hits"]["hits"][i]["_source"][item])
                return_object[item] = item_list
            return_object["score"] = [
                results["hits"]["hits"][i]["_score"]
                for i in range(len(results["hits"]["hits"]))
            ]
        return return_object
