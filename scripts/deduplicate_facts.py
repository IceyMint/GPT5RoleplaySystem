import argparse
import asyncio
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import List

from neo4j import GraphDatabase

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gpt5_roleplay_system.config import load_config
from gpt5_roleplay_system.llm import OpenRouterLLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deduplicate_facts")

DEDUPLICATION_SYSTEM_PROMPT = "You are a precision data cleaner."
DEDUPLICATION_USER_PROMPT = (
    "You are a data cleaning expert. Refine these facts about a person, removing redundancy and duplicates. "
    "Combine similar facts. Output a JSON object with key 'facts' containing a list of strings.\n\n"
    "Person Name: {name}\nCurrent Facts: {facts}"
)

async def deduplicate_person_facts(llm_client: OpenRouterLLMClient, name: str, facts: List[str]) -> List[str]:
    if not facts:
        return []
    
    prompt = DEDUPLICATION_USER_PROMPT.format(name=name, facts=json.dumps(facts))
    
    try:
        response_text = await asyncio.to_thread(llm_client._request_text, DEDUPLICATION_SYSTEM_PROMPT, prompt)

        clean_text = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", response_text, flags=re.DOTALL)
        start = clean_text.find("{")
        end = clean_text.rfind("}")
        if start != -1 and end != -1:
            clean_text = clean_text[start : end + 1]

        data = json.loads(clean_text)
        refined_raw = data.get("facts", [])
        if not isinstance(refined_raw, list):
            raise ValueError("facts must be a list")

        refined_facts: List[str] = []
        seen: set[str] = set()
        for item in refined_raw:
            if not isinstance(item, str):
                continue
            fact = item.strip()
            if not fact or fact in seen:
                continue
            seen.add(fact)
            refined_facts.append(fact)
        return refined_facts
    except Exception as e:
        logger.error(f"Error deduplicating facts for {name}: {e}")
        return facts

def _config_path_from_args(path: str | None) -> str:
    if path:
        return path
    default_path = REPO_ROOT / "config.yaml"
    if default_path.exists():
        return str(default_path)
    return ""


async def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate person facts in Neo4j.")
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: repo-root config.yaml).")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all people with >1 fact instead of only needs_dedupe=true.",
    )
    args = parser.parse_args()

    config_path = _config_path_from_args(args.config)
    config = load_config(config_path if config_path else None)
    neo4j_conf = config.neo4j
    llm_conf = config.llm

    llm_client = OpenRouterLLMClient(
        api_key=llm_conf.api_key,
        base_url=llm_conf.base_url,
        model=llm_conf.model,
        address_model=llm_conf.address_model,
        max_tokens=llm_conf.max_tokens,
        temperature=llm_conf.temperature,
        timeout_seconds=llm_conf.timeout_seconds,
        reasoning=llm_conf.reasoning,
    )

    driver = GraphDatabase.driver(neo4j_conf.uri, auth=(neo4j_conf.user, neo4j_conf.password))
    database = neo4j_conf.database

    logger.info(f"Connecting to Neo4j database: {database}")

    if args.all:
        query = (
            "MATCH (p:Person) "
            "WHERE size(coalesce(p.facts, [])) > 1 "
            "RETURN p.user_id as user_id, p.name as name, p.facts as facts"
        )
    else:
        query = (
            "MATCH (p:Person) "
            "WHERE size(coalesce(p.facts, [])) > 1 AND coalesce(p.needs_dedupe, false) = true "
            "RETURN p.user_id as user_id, p.name as name, p.facts as facts"
        )

    with driver.session(database=database) as session:
        result = session.run(query)
        people = list(result)

    logger.info(f"Found {len(people)} people with facts to process.")

    for person in people:
        user_id = person["user_id"]
        name = person["name"]
        facts = [item for item in (person["facts"] or []) if isinstance(item, str) and item.strip()]

        if len(facts) < 2:
            continue

        logger.info(f"Processing {name} ({user_id}) with {len(facts)} facts...")
        refined_facts = await deduplicate_person_facts(llm_client, name, facts)
        deduped_ts = time.time()

        if refined_facts and refined_facts != facts:
            logger.info(f"Refined {len(facts)} facts down to {len(refined_facts)} for {name}.")
            with driver.session(database=database) as session:
                session.run(
                    "MATCH (p:Person {user_id: $user_id}) "
                    "SET p.facts = $facts, "
                    "    p.needs_dedupe = false, "
                    "    p.facts_deduped_ts = $facts_deduped_ts",
                    user_id=user_id,
                    facts=refined_facts,
                    facts_deduped_ts=deduped_ts,
                )
        else:
            logger.info(f"No changes for {name}.")
            with driver.session(database=database) as session:
                session.run(
                    "MATCH (p:Person {user_id: $user_id}) "
                    "SET p.needs_dedupe = false, "
                    "    p.facts_deduped_ts = $facts_deduped_ts",
                    user_id=user_id,
                    facts_deduped_ts=deduped_ts,
                )

    driver.close()
    logger.info("Deduplication complete.")

if __name__ == "__main__":
    asyncio.run(main())
