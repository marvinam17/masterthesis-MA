import re
import json
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from SPARQLWrapper import SPARQLWrapper, JSON


def convert_beir_dbpedia_entity_v2_to_df(corpus:dict):
    """
    Convert the BEIR DBPedia Entity V2 corpus to a Pandas DataFrame.

    Args:
        corpus (dict): The BEIR DBPedia Entity V2 corpus.

    Returns:
        pd.DataFrame: The converted DataFrame.
    """
    df = pd.DataFrame.from_dict(corpus, orient='index')
    df = df.reset_index()
    df = df.rename(columns={'index': 'id'})
    df = df.drop(df[~df["id"].str.contains("dbpedia:")].index)
    df = df.drop(df[df["id"]=="<dbpedia:List_of_fictional_extraterrestrials_by_form>"].index)
    df.reset_index(drop=True, inplace=True)
    return df

def add_wikipedia_page_id(df:pd.DataFrame, mapping_df:pd.DataFrame):
    """
    Add the Wikipedia page ID to the DataFrame.
    First uses the dbpedia dump mapping, then tries to scrape the actual dbpedia page for the page ID.

    Args:
        df (pd.DataFrame): The DataFrame to add the Wikipedia page ID to.

    Returns:
        pd.DataFrame: The DataFrame with the Wikipedia page ID added.
    """
    df = df.merge(mapping_df,how="left",on="id")
    page_map = {}
    for _, row in df[df["page_id"].isna()][["id","page_id"]].iterrows():
        try:
            response = requests.get("https://dbpedia.org/page/" + row["id"][9:-1])
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                for meta in soup.find_all("span"):
                    if meta.get("property") == "dbo:wikiPageID":
                        page_map[row["id"]] = int(meta.get_text())
                if row["id"] not in page_map:
                    print("No page id found for: " + row["id"])
        except:
            print("Error for: " + row["id"])
            continue
    # Manually Add four items that are not in the mapping:
    # <dbpedia:Beta_(programming_language)> -> https://dbpedia.org/page/BETA_(programming_language) -> 135868 
    page_map["<dbpedia:Beta_(programming_language)>"] = 135868
    # <dbpedia:Hooligans_(Film)> -> http://dbpedia.org/resource/Hooligans_(film)> -> 31396323
    page_map["<dbpedia:Hooligans_(Film)>"] = 31396323
    # <dbpedia:Snow_Queen_(Vinge_novel)> -> https://dbpedia.org/page/The_Snow_Queen_(Vinge_novel) -> 1882707
    page_map["<dbpedia:Snow_Queen_(Vinge_novel)>"] = 1882707
    # <dbpedia:Hooligans_(Film)> -> https://dbpedia.org/page/Arm_in_Arm_Down_the_Street_(1966_film) -> 9919707
    page_map["<dbpedia:Arm_in_Arm_Down_the_Street_(1966_fim)>"] = 9919707 
    df.loc[df['page_id'].isnull(),'page_id'] = df['id'].map(page_map)
    # Convert to integer
    df["page_id"] = df["page_id"].astype(int)
    return df
 

def get_wikidata_id_from_dbpedia_dump(path:str="../03_data/raw_data/dbpedia/sameas_all_wikis_wikidata.ttl") -> dict:
    """
    Get the mapping from DBPedia to Wikidata from the DBPedia dump.
    WikidataID starts with Q and is followed by a number.
    """
    with open(path) as file:
        lines = file.readlines()
    mapping_dbpedia_wikidata_ids ={}
    for line in lines[1:-1]:
        splitted = line.split(" ")
        if splitted[2].startswith("<http://dbpedia.org/resource/"):
            id = splitted[2].replace("http://dbpedia.org/resource/","dbpedia:")
            item_id = int(re.search("Q\d{1,10}",splitted[0]).group(0)[1:])
            mapping_dbpedia_wikidata_ids[id] = item_id
    return mapping_dbpedia_wikidata_ids



def execute_sparql_query(query: str, endpoint: str = "https://dbpedia.org/sparql") -> list:
    """
    Execute a SPARQL query and return the results.
    
    Args:
        query (str): A SPARQL query.
        endpoint (str): The SPARQL endpoint to use. Default is the DBPedia endpoint.

    Returns:
        list: A list of dictionaries containing the results.
    """
    try:
        # Initialisiere den SPARQL-Wrapper mit dem Endpunkt
        sparql = SPARQLWrapper(endpoint)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)  # Ergebnisse im JSON-Format
        
        # Führe die Abfrage aus und hole die Ergebnisse
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        print(f"Fehler beim Ausführen der SPARQL-Query: {e}")
        return []
    
def postprocess_results(items:list):
    """
    Add the results of a SPARQL query to a dictionary.
    Args:
        items (list): A list of dictionaries containing the results of a SPARQL query.

    Returns:
        dict: A dictionary containing the mapping from Wikipedia page ID to Wikidata ID.
    """
    return_obj = {}
    for item in items:
        if item["wikidata_concept"] and item["wikiPageID"]:
            return_obj[int(item["wikiPageID"]["value"])] = int(item["wikidata_concept"]["value"].replace("http://www.wikidata.org/entity/Q",""))
    return return_obj

def batch(iterable, n=1):
    """
    Batch an iterable into chunks of size n.
    Taken from https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks

    Args:
        iterable: The iterable to batch.
        n: The size of the batches.
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def add_wikidata_id(df:pd.DataFrame, path:str="../03_data/raw_data/dbpedia/sameas_all_wikis_wikidata.ttl", save_sparql_mapping: bool = False) -> pd.DataFrame:
    """
    Add the Wikidata ID to the DataFrame.
    First uses the mapping from the DBPedia dump, then tries to get the Wikidata ID via SPARQL.
    SPARQL is used for the remaining items that are not in the mapping.
    Iterated in Batches

    Args:
        df (pd.DataFrame): The DataFrame to add the Wikidata ID to.

    Returns:
        pd.DataFrame: The DataFrame with the Wikidata ID added.
    """
    mapping_dbpedia_wikidata_ids = get_wikidata_id_from_dbpedia_dump(path)
    df["item_id"] = df["id"].map(mapping_dbpedia_wikidata_ids)
    not_existing_relations = df[df["item_id"].isna()]["page_id"].astype(str).to_list()
    results = []
    for x in batch(not_existing_relations, 100):
        page_ids = " ".join(x)
        sparql_query = f"""
            SELECT ?wikidata_concept ?wikiPageID 
            WHERE {{
                ?resource dbo:wikiPageID ?wikiPageID .
                ?resource owl:sameAs ?wikidata_concept .
                FILTER(CONTAINS(STR(?wikidata_concept), "wikidata.org"))
                VALUES ?wikiPageID {{ {page_ids} }}
            }}
            LIMIT 100
            """
        results.extend(execute_sparql_query(sparql_query))
    sparql_mapping = postprocess_results(results)
    if save_sparql_mapping:
        with open('../03_data/preprocessed_data/sparql_mapping_dbpedia_wikidata.json', 'w') as f:
            json.dump(sparql_mapping, f)
    df["item_id_sparql"] = df["page_id"].map(sparql_mapping)
    df["item_id"] = df["item_id"].fillna(df["item_id_sparql"])
    df = df.drop(columns=["item_id_sparql"],axis=1)
    return df   
    
def df_to_unique_permutations(df):
    """
    Konvertiert einen DataFrame mit zwei Spalten in ein 2D-Numpy-Array und enthält nur einzigartige Permutationen.

    Args:
        df (pd.DataFrame): DataFrame mit zwei Spalten.

    Returns:
        np.ndarray: 2D-Numpy-Array mit einzigartigen Permutationen.
    """
    # Konvertiere den DataFrame in ein Numpy-Array
    array = df.to_numpy()
    
    # Erstelle eine Liste von einzigartigen Permutationen
    unique_permutations = set()
    for row in array:
        unique_permutations.add(tuple(sorted(row)))
    
    # Konvertiere die Menge der einzigartigen Permutationen zurück in ein Numpy-Array
    unique_array = np.array(list(unique_permutations))
    
    return unique_array

def preprocess_wikiPageLinks(df:pd.DataFrame, path:str):
    """
    Preprocess the page links unredirected file from DBPedia.
    Removes all realtions that lead from one node to itself.
    Removes all permutations of relations that are already in the DataFrame.
    This results in an undirected graph. And a maximum of one relation between two nodes.

    Args:
        df (pd.DataFrame): The DataFrame containing the nodes.
    Returns:
        pd.DataFrame: The DataFrame containing the relations.
    """
    with open(path) as file:
        lines = file.readlines()

    real_rels = []
    existing_pages = set(df.id.to_list())
    for line in lines: 
        splitted  = line.split(" ")
        source = splitted[0].replace("http://dbpedia.org/resource/","dbpedia:")
        if source in existing_pages:
            target = splitted[2].replace("http://dbpedia.org/resource/","dbpedia:")
            if target in existing_pages:
                real_rels.append({
                    "source": source,
                    "target": target
                })

    real_rels = pd.DataFrame(real_rels)
    real_rels = real_rels[real_rels["source"]!=real_rels["target"]]
    item_to_index = df["id"].to_dict()
    inv_item_to_index = {v: k for k, v in item_to_index.items()}
    real_rels["source"] = real_rels["source"].map(inv_item_to_index)
    real_rels["target"] = real_rels["target"].map(inv_item_to_index)
    np_rels = df_to_unique_permutations(real_rels)
    return pd.DataFrame(np_rels, columns=["source", "target"])    


def preprocess_wikidataLinks(df:pd.DataFrame, path:str):
    """
    Preprocess the wikidata links unredirected file from DBPedia.
    Removes all relations that lead from one node to itself.
    Removes all permutations of relations that are already in the DataFrame.
    This results in an undirected graph. And a maximum of one relation between two nodes.

    Args:
        df (pd.DataFrame): The DataFrame containing the nodes.
    Returns:
        pd.DataFrame: The DataFrame containing the relations.
    """
    with open(path) as file:
        lines = file.readlines()

    existing_pages = set(df.item_id.dropna().astype(int).to_list())
    real_rels = []
    for line in lines[1:-1]:
        splitted  = line.split(" ")
        if splitted[0].startswith("<http://wikidata.dbpedia.org/") and splitted[2].startswith("<http://wikidata.dbpedia.org/"):
            source = int(splitted[0][39:-1])
            if source in existing_pages :
                target = int(splitted[2][39:-1])
                if target in existing_pages:
                    real_rels.append({
                        "source": source,
                        "target": target
                    })

    real_rels = pd.DataFrame(real_rels)
    real_rels = real_rels[real_rels["source"]!=real_rels["target"]]
    item_to_index = df.item_id.dropna().astype(int).to_dict()
    inv_item_to_index = {v: k for k, v in item_to_index.items()}
    real_rels["source"] = real_rels["source"].map(inv_item_to_index)
    real_rels["target"] = real_rels["target"].map(inv_item_to_index)
    np_rels = df_to_unique_permutations(real_rels)
    return pd.DataFrame(np_rels, columns=["source", "target"])    
