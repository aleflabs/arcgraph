CREATE (n:Person {neo4j_id: "1"});
CREATE (m:Person {neo4j_id: "2"});
MATCH (a),(b) WHERE a.neo4j_id="1" AND b.neo4j_id="2" CREATE (a)-[:KNOWS]->(b);
