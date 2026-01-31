import asyncio
import json
import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase
from gpt5_roleplay_system.config import load_config
from gpt5_roleplay_system.llm import OpenRouterLLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deduplicate_facts")

DEDUPLICATION_SYSTEM_PROMPT = """
You are a data cleaning expert. Your task is to deduplicate and refine a list of facts about a person in a roleplay system.
The facts might be redundant, slightly different versions of the same information, or contradictory.

# GUIDELINES:
1. Combine similar facts into a single, concise statement.
2. Remove redundant information.
3. Keep the most detailed version of a fact if there are multiple versions.
4. Ensure the resulting facts are "durable" (long-term information).
5. Output the result as a JSON object with a single key "facts" containing the list of strings.

# INPUT:
Person Name: {name}
Current Facts: {facts}

# OUTPUT FORMAT:
{{
  "facts": ["refined fact 1", "refined fact 2", ...]
}}
"""

async def deduplicate_person_facts(llm_client: OpenRouterLLMClient, name: str, facts: List[str]) -> List[str]:
    if not facts:
        return []
    
    prompt = DEDUPLICATION_SYSTEM_PROMPT.format(name=name, facts=json.dumps(facts))
    
    try:
        # We use _request_text for simplicity or we could use structured output if we want to be more formal
        # Using a raw request here to ensure we get exactly what we want
        response_text = await asyncio.to_thread(llm_client._request_text, "You are a helpful assistant.", prompt)
        
        # Basic cleanup of LLM response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        data = json.loads(response_text)
        refined_facts = data.get("facts", [])
        return refined_facts
    except Exception as e:
        logger.error(f"Error deduplicating facts for {name}: {e}")
        return facts

async def main():
    config = load_config()
    neo4j_conf = config.neo4j
    llm_conf = config.llm

    llm_client = OpenRouterLLMClient(
        api_key=llm_conf.api_key,
        base_url=llm_conf.base_url,
        model=llm_conf.model,
        max_tokens=2048,
        temperature=0.2
    )

    driver = GraphDatabase.driver(neo4j_conf.uri, auth=(neo4j_conf.user, neo4j_conf.password))
    database = neo4j_conf.database

    logger.info(f"Connecting to Neo4j database: {database}")

    with driver.session(database=database) as session:
        result = session.run("MATCH (p:Person) WHERE size(p.facts) > 0 RETURN p.user_id as user_id, p.name as name, p.facts as facts")
        people = list(result)

    logger.info(f"Found {len(people)} people with facts to process.")

    for person in people:
        user_id = person["user_id"]
        name = person["name"]
        facts = person["facts"]
        
        if len(facts) < 2:
            continue

        logger.info(f"Processing {name} ({user_id}) with {len(facts)} facts...")
        refined_facts = await deduplicate_person_facts(llm_client, name, facts)
        
        if refined_facts and refined_facts != facts:
            logger.info(f"Refined {len(facts)} facts down to {len(refined_facts)} for {name}.")
            with driver.session(database=database) as session:
                session.run(
                    "MATCH (p:Person {user_id: $user_id}) SET p.facts = $facts",
                    user_id=user_id,
                    facts=refined_facts
                )
        else:
            logger.info(f"No changes for {name}.")

    driver.close()
    logger.info("Deduplication complete.")

if __name__ == "__main__":
    asyncio.run(main())
