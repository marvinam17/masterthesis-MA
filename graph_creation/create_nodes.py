import pandas as pd
from utils import Neo4jConnection

def create_query_list(df):
    """
    Diese Funktion erstellt eine Liste von Cypher Queries, die einen Knoten je
    Zeile eines Dataframes erstellt.
    """
    queries = []
    for _, row in df.iterrows():
        query = f"""CREATE (p:Page {{name: {row['title']}, item_id: {row['item_id']}, page_id: {row['page_id']}}})"""
        queries.append(query)
    return queries

if __name__ == '__main__':
    # Page Daten einlesen
    pages = pd.read_csv("../raw_data/page.csv")
    # Cypher Queries generieren
    queries = create_query_list(pages)
    # Connection einrichten 
    conn = Neo4jConnection("bolt://localhost:7687","neo4j","academic")
    # Queries iterativ ausführen
    for query in queries:
        conn.run_query(query)
        break