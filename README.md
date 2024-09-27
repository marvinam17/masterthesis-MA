# Masterthesis

## Topic:

## Author:

## USEFULL CYPHER QUERIES
 - Source: https://neo4j.com/docs/getting-started/data-import/csv-import/

### Create page nodes

#### Index on item_id for relation creation
```
CREATE INDEX item_index FOR (p:Page) ON (p.item_id);
```

```
:auto LOAD CSV WITH HEADERS FROM 'file:///page.csv' AS line
CALL {
WITH line
CREATE (p:Page {page_id: toInteger(line.page_id), item_id: toInteger(line.item_id), title: line.title, views: toInteger(line.views)})
} IN TRANSACTIONS OF 50000 ROWS;
```

### Create Wikimedia Relations

```
:auto LOAD CSV WITH HEADERS FROM 'file:///statements.csv' AS row
CALL {
WITH row
MATCH (p1:Page {item_id: toInteger(row.source_item_id)}), (p2:Page {item_id: toInteger(row.target_item_id)})
CREATE (p1)-[:RELATION {property_id: toInteger(row.property_id)}]->(p2)}
IN TRANSACTIONS OF 100000 ROWS;
```

### Index on relations
```
CREATE INDEX composite_range_rel_index_relation FOR ()-[r:RELATION]-() ON (r.property_id)
```





:auto LOAD CSV WITH HEADERS FROM 'file:///statements.csv' AS row
CALL {
WITH row LIMIT 1000
MATCH (p1:Page {item_id: toInteger(row.source_item_id)}), (p2:Page {item_id: toInteger(row.target_item_id)})
CREATE (p1)-[:RELATION {property_id: toInteger(row.edge_property_id)}]->(p2)}
IN TRANSACTIONS OF 100000 ROWS;





### Delete all relationships

CALL apoc.periodic.iterate(
"MATCH (:Page)-[r:RELATION]->(:Page) RETURN r",
"DELETE r",
{batchSize:10000})