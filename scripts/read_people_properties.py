import sys
import os
from neo4j import GraphDatabase

def main():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "suprgurl265"
    database = "gpt5-roleplay"

    print(f"Connecting to Neo4j at {uri} (database: {database})...")
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            result = session.run("MATCH (p:Person) RETURN p")
            records = list(result)
            
            if not records:
                print("No Person nodes found in the database.")
                return

            print(f"Found {len(records)} Person nodes:\n")
            for record in records:
                person = record["p"]
                print(f"--- Person: {person.get('name', 'N/A')} ---")
                for key, value in person.items():
                    print(f"  {key}: {value}")
                print()

        driver.close()
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
