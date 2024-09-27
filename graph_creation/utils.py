from neo4j import GraphDatabase

class Neo4jConnection:

    def __init__(self, uri, user, password):
        """Initialisiert die Verbindung zur Neo4j-Datenbank."""
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password), max_connection_lifetime=200)
            print("Verbindung zur Neo4j-Datenbank erfolgreich hergestellt.")
        except Exception as e:
            print(f"Verbindung fehlgeschlagen: {e}")

    def close(self):
        """Schließt die Verbindung zur Datenbank."""
        if self.driver:
            self.driver.close()
            print("Verbindung zur Neo4j-Datenbank geschlossen.")

    def run_query(self, query, parameters=None):
        """Führt eine Cypher-Abfrage aus."""
        if not self.driver:
            raise Exception("Keine aktive Verbindung zur Datenbank.")
        
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
        
    def create_edges(self, edges):
        query = """
            UNWIND $edges AS edge
            MATCH (source:Page {item_id: edge.source_item_id})
            MATCH (target:Page {item_id: edge.target_item_id})
            MERGE (source)-[r:{edge.en_label}]->(target)
            SET r.property_id={edge.edge_property_id}, r.description={edge.en_description}
            RETURN source, r, target
        """
        with self.driver.session() as session:
            result = session.run(query, edges=edges)
            return result.data()