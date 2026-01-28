# Fresh Neo4j DB Setup (GPT5RoleplaySystem)

The server now supports a named Neo4j database via `database.name` in `config.yaml` (or `NEO4J_DATABASE`).

Default in this repo:
- `database.name: gpt5-roleplay`

The code will *attempt* to create the database and constraints automatically, but for a clean start it's best to run this once with admin rights.

## Cypher-shell (recommended)

```bash
cypher-shell -u neo4j -p 'YOUR_PASSWORD'
```

Then run:

```cypher
:use system
CREATE DATABASE gpt5-roleplay IF NOT EXISTS;

:use gpt5-roleplay
CREATE CONSTRAINT person_user_id_unique IF NOT EXISTS
FOR (p:Person) REQUIRE p.user_id IS UNIQUE;

CREATE CONSTRAINT person_name_lower_unique IF NOT EXISTS
FOR (p:Person) REQUIRE p.name_lower IS UNIQUE;

CREATE INDEX person_name_lower_idx IF NOT EXISTS
FOR (p:Person) ON (p.name_lower);

CREATE CONSTRAINT experience_id_unique IF NOT EXISTS
FOR (e:Experience) REQUIRE e.id IS UNIQUE;

CREATE VECTOR INDEX experience_embedding_idx IF NOT EXISTS
FOR (e:Experience) ON (e.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
```

## Notes

- If you already have duplicate `name_lower` values, the `person_name_lower_unique` constraint may fail.
- Uniqueness for merges should be based on `user_id` (UUID). Names should be treated as aliases/display values.
- Adjust vector dimensions if you change embedding models.
- For Neo4j GenAI embeddings, set `OPENAI_API_KEY` in the environment (or `api_keys.openai_api_key` in config).
- Experience nodes are created from episodic summaries (see `episode_summary` in `config.yaml`).
