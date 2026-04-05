from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
import unicodedata
from typing import Any, Dict, List

from .cypher_utils import quote_cypher_identifier

logger = logging.getLogger(__name__)


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for item in items:
        key = _normalize_fact_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _normalize_fact_key(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    cleaned = "".join(ch if ch.isalnum() else " " for ch in normalized)
    return " ".join(cleaned.split())


@dataclass
class PersonProfile:
    user_id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    last_seen_ts: float = 0.0


@dataclass
class PersonNameIndexEntry:
    user_id: str
    names: List[str] = field(default_factory=list)


class KnowledgeStore:
    def fetch_people(self, user_ids: List[str]) -> Dict[str, PersonProfile]:
        raise NotImplementedError

    def fetch_people_by_name(self, names: List[str]) -> Dict[str, PersonProfile]:
        raise NotImplementedError

    def fetch_people_by_partial_name(self, names: List[str]) -> Dict[str, PersonProfile]:
        raise NotImplementedError

    def fetch_people_name_index(self) -> List[PersonNameIndexEntry]:
        raise NotImplementedError

    def upsert_person_facts(self, user_id: str, name: str, facts: List[str]) -> None:
        raise NotImplementedError

    def upsert_relationship(self, source_id: str, target_id: str, relationship: str) -> None:
        raise NotImplementedError

    def update_last_seen(self, user_id: str, name: str, timestamp: float) -> None:
        raise NotImplementedError


class InMemoryKnowledgeStore(KnowledgeStore):
    def __init__(self) -> None:
        self._people: Dict[str, PersonProfile] = {}

    @staticmethod
    def _merge_aliases(existing: List[str], candidates: List[str]) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for value in list(existing or []) + list(candidates or []):
            text = str(value or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(text)
        return merged

    def fetch_people(self, user_ids: List[str]) -> Dict[str, PersonProfile]:
        return {user_id: self._people[user_id] for user_id in user_ids if user_id in self._people}

    def fetch_people_by_name(self, names: List[str]) -> Dict[str, PersonProfile]:
        if not names:
            return {}
        names_lower = {name.casefold() for name in names if name}
        results: Dict[str, PersonProfile] = {}
        for profile in self._people.values():
            profile_names = [profile.name] + list(profile.aliases or [])
            if any(str(value or "").casefold() in names_lower for value in profile_names):
                results[profile.user_id] = profile
        return results

    def fetch_people_by_partial_name(self, names: List[str]) -> Dict[str, PersonProfile]:
        if not names:
            return {}
        tokens = [name.casefold() for name in names if name]
        results: Dict[str, PersonProfile] = {}
        for profile in self._people.values():
            profile_names = [profile.name] + list(profile.aliases or [])
            profile_names_lower = [str(value or "").casefold() for value in profile_names if value]
            if any(token in name_value for token in tokens for name_value in profile_names_lower):
                results[profile.user_id] = profile
        return results

    def fetch_people_name_index(self) -> List[PersonNameIndexEntry]:
        entries: List[PersonNameIndexEntry] = []
        for profile in self._people.values():
            if not profile.user_id:
                continue
            names = self._merge_aliases([], [profile.name] + list(profile.aliases or []))
            if not names:
                continue
            entries.append(PersonNameIndexEntry(user_id=profile.user_id, names=names))
        return entries

    def upsert_person_facts(self, user_id: str, name: str, facts: List[str]) -> None:
        profile = self._people.get(user_id) or PersonProfile(user_id=user_id, name=name)
        if name:
            profile.name = name
        profile.aliases = self._merge_aliases(profile.aliases, [name, profile.name])
        existing_keys = {_normalize_fact_key(item) for item in profile.facts if _normalize_fact_key(item)}
        for fact in facts:
            key = _normalize_fact_key(fact)
            if not key or key in existing_keys:
                continue
            existing_keys.add(key)
            profile.facts.append(fact)
        self._people[user_id] = profile

    def upsert_relationship(self, source_id: str, target_id: str, relationship: str) -> None:
        source = self._people.get(source_id) or PersonProfile(user_id=source_id, name=source_id)
        source.relationships.append({"target_id": target_id, "relationship": relationship})
        self._people[source_id] = source

    def update_last_seen(self, user_id: str, name: str, timestamp: float) -> None:
        if not user_id:
            return
        profile = self._people.get(user_id) or PersonProfile(user_id=user_id, name=name or user_id)
        if name:
            profile.name = name
        profile.aliases = self._merge_aliases(profile.aliases, [name, profile.name])
        profile.last_seen_ts = max(float(timestamp or 0.0), float(profile.last_seen_ts or 0.0))
        self._people[user_id] = profile


class Neo4jKnowledgeStore(KnowledgeStore):
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise RuntimeError("neo4j package is not installed") from exc
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database or "neo4j"
        self._ensure_database_and_constraints()

    def _system_session(self):
        return self._driver.session(database="system")

    def _session(self):
        return self._driver.session(database=self._database)

    def _ensure_database_and_constraints(self) -> None:
        # Try to create the target database first.
        try:
            with self._system_session() as system_session:
                system_session.run(f"CREATE DATABASE {quote_cypher_identifier(self._database)} IF NOT EXISTS")
        except Exception as exc:
            logger.warning("Neo4j schema setup: unable to ensure database %s (%s)", self._database, exc)

        constraint_statements = [
            "CREATE CONSTRAINT person_user_id_unique IF NOT EXISTS "
            "FOR (p:Person) REQUIRE p.user_id IS UNIQUE",
            # name_lower is useful for exact matches and dedupe; may fail if duplicates exist.
            "CREATE CONSTRAINT person_name_lower_unique IF NOT EXISTS "
            "FOR (p:Person) REQUIRE p.name_lower IS UNIQUE",
            "CREATE CONSTRAINT experience_id_unique IF NOT EXISTS "
            "FOR (e:Experience) REQUIRE e.id IS UNIQUE",
        ]
        index_statements = [
            "CREATE INDEX person_name_lower_idx IF NOT EXISTS FOR (p:Person) ON (p.name_lower)",
            "CREATE INDEX experience_sender_idx IF NOT EXISTS FOR (e:Experience) ON (e.sender_id)",
        ]

        with self._session() as session:
            for statement in constraint_statements + index_statements:
                try:
                    session.run(statement)
                except Exception as exc:
                    logger.warning("Neo4j schema setup failed for statement [%s]: %s", statement, exc)

    def fetch_people(self, user_ids: List[str]) -> Dict[str, PersonProfile]:
        if not user_ids:
            return {}
        query = """
        MATCH (p:Person)
        WHERE p.user_id IN $user_ids
        OPTIONAL MATCH (p)-[r]->(o:Person)
        WITH p, collect(
            CASE
                WHEN o IS NULL OR r IS NULL THEN NULL
                ELSE {target_id: o.user_id, relationship: type(r)}
            END
        ) AS rels
        RETURN p, [rel IN rels WHERE rel IS NOT NULL AND rel.target_id IS NOT NULL] AS rels
        """
        profiles: Dict[str, PersonProfile] = {}
        with self._session() as session:
            for record in session.run(query, user_ids=user_ids):
                person = record["p"]
                rels = record["rels"]
                profile = PersonProfile(
                    user_id=person.get("user_id"),
                    name=person.get("name", ""),
                    aliases=_dedupe_preserve_order(person.get("aliases", []) or []),
                    facts=_dedupe_preserve_order(person.get("facts", []) or []),
                    relationships=rels or [],
                    last_seen_ts=float(person.get("last_seen_ts", 0.0) or 0.0),
                )
                profiles[profile.user_id] = profile
        return profiles

    def fetch_people_by_name(self, names: List[str]) -> Dict[str, PersonProfile]:
        if not names:
            return {}
        query = """
        MATCH (p:Person)
        WHERE coalesce(p.name_lower, toLower(p.name)) IN $names
           OR any(alias IN coalesce(p.aliases, []) WHERE toLower(alias) IN $names)
        OPTIONAL MATCH (p)-[r]->(o:Person)
        WITH p, collect(
            CASE
                WHEN o IS NULL OR r IS NULL THEN NULL
                ELSE {target_id: o.user_id, relationship: type(r)}
            END
        ) AS rels
        RETURN p, [rel IN rels WHERE rel IS NOT NULL AND rel.target_id IS NOT NULL] AS rels
        """
        profiles: Dict[str, PersonProfile] = {}
        names_lower = [name.lower() for name in names if name]
        with self._session() as session:
            for record in session.run(query, names=names_lower):
                person = record["p"]
                rels = record["rels"]
                profile = PersonProfile(
                    user_id=person.get("user_id"),
                    name=person.get("name", ""),
                    aliases=_dedupe_preserve_order(person.get("aliases", []) or []),
                    facts=_dedupe_preserve_order(person.get("facts", []) or []),
                    relationships=rels or [],
                    last_seen_ts=float(person.get("last_seen_ts", 0.0) or 0.0),
                )
                profiles[profile.user_id] = profile
        return profiles

    def fetch_people_by_partial_name(self, names: List[str]) -> Dict[str, PersonProfile]:
        if not names:
            return {}
        names_lower = [name.lower() for name in names if name]
        query = """
        MATCH (p:Person)
        WHERE any(token IN $names WHERE coalesce(p.name_lower, toLower(p.name)) CONTAINS token)
           OR any(token IN $names WHERE any(alias IN coalesce(p.aliases, []) WHERE toLower(alias) CONTAINS token))
        OPTIONAL MATCH (p)-[r]->(o:Person)
        WITH p, collect(
            CASE
                WHEN o IS NULL OR r IS NULL THEN NULL
                ELSE {target_id: o.user_id, relationship: type(r)}
            END
        ) AS rels
        RETURN p, [rel IN rels WHERE rel IS NOT NULL AND rel.target_id IS NOT NULL] AS rels
        """
        profiles: Dict[str, PersonProfile] = {}
        with self._session() as session:
            for record in session.run(query, names=names_lower):
                person = record["p"]
                rels = record["rels"]
                profile = PersonProfile(
                    user_id=person.get("user_id"),
                    name=person.get("name", ""),
                    aliases=_dedupe_preserve_order(person.get("aliases", []) or []),
                    facts=_dedupe_preserve_order(person.get("facts", []) or []),
                    relationships=rels or [],
                    last_seen_ts=float(person.get("last_seen_ts", 0.0) or 0.0),
                )
                profiles[profile.user_id] = profile
        return profiles

    def fetch_people_name_index(self) -> List[PersonNameIndexEntry]:
        query = """
        MATCH (p:Person)
        WHERE p.user_id IS NOT NULL AND trim(p.user_id) <> ""
        RETURN p.user_id AS user_id,
               p.name AS name,
               coalesce(p.aliases, []) AS aliases
        """
        entries: List[PersonNameIndexEntry] = []
        with self._session() as session:
            for record in session.run(query):
                user_id = str(record.get("user_id") or "").strip()
                if not user_id:
                    continue
                names: List[str] = []
                seen: set[str] = set()
                for value in [record.get("name", "")] + list(record.get("aliases", []) or []):
                    text = str(value or "").strip()
                    if not text:
                        continue
                    key = text.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    names.append(text)
                if names:
                    entries.append(PersonNameIndexEntry(user_id=user_id, names=names))
        return entries

    @staticmethod
    def _name_lower_owner_user_id(session: Any, name_lower: str) -> str:
        if not name_lower:
            return ""
        result = session.run(
            "MATCH (p:Person {name_lower: $name_lower}) "
            "RETURN coalesce(p.user_id, '') AS user_id "
            "LIMIT 1",
            name_lower=name_lower,
        )
        record = result.single()
        if not record:
            return ""
        return str(record.get("user_id", "") or "")

    def upsert_person_facts(self, user_id: str, name: str, facts: List[str]) -> None:
        name = name or ""
        name_lower = name.strip().lower()
        facts = _dedupe_preserve_order([fact for fact in facts if fact])
        has_new_facts = bool(facts)
        facts_updated_ts = time.time() if has_new_facts else 0.0
        alias_values = [name] if name else []
        claim_by_name = """
        MATCH (p:Person {name_lower: $name_lower})
        WHERE $name_lower <> "" AND (p.user_id IS NULL OR p.user_id = "")
        SET p.user_id = $user_id,
            p.name = CASE WHEN $name <> "" THEN $name ELSE p.name END,
            p.name_lower = CASE WHEN $name_lower <> "" THEN $name_lower ELSE p.name_lower END,
            p.aliases = reduce(acc = [], x IN coalesce(p.aliases, []) + $aliases |
                CASE WHEN x IS NULL OR trim(x) = "" OR x IN acc THEN acc ELSE acc + x END),
            p.facts = reduce(acc = [], x IN coalesce(p.facts, []) + $facts |
                CASE WHEN x IS NULL OR trim(x) = "" OR x IN acc THEN acc ELSE acc + x END),
            p.needs_dedupe = CASE WHEN $has_new_facts THEN true ELSE coalesce(p.needs_dedupe, false) END,
            p.facts_updated_ts = CASE
                WHEN $has_new_facts THEN $facts_updated_ts
                ELSE coalesce(p.facts_updated_ts, 0.0)
            END
        RETURN p
        """
        merge_by_id = """
        MERGE (p:Person {user_id: $user_id})
        SET p.name = CASE WHEN $name <> "" THEN $name ELSE p.name END,
            p.name_lower = CASE
                WHEN $set_name_lower AND $name_lower <> "" THEN $name_lower
                ELSE p.name_lower
            END,
            p.aliases = reduce(acc = [], x IN coalesce(p.aliases, []) + $aliases |
                CASE WHEN x IS NULL OR trim(x) = "" OR x IN acc THEN acc ELSE acc + x END),
            p.facts = reduce(acc = [], x IN coalesce(p.facts, []) + $facts |
                CASE WHEN x IS NULL OR trim(x) = "" OR x IN acc THEN acc ELSE acc + x END),
            p.needs_dedupe = CASE WHEN $has_new_facts THEN true ELSE coalesce(p.needs_dedupe, false) END,
            p.facts_updated_ts = CASE
                WHEN $has_new_facts THEN $facts_updated_ts
                ELSE coalesce(p.facts_updated_ts, 0.0)
            END
        """
        with self._session() as session:
            conflicting_owner = ""
            if user_id and name_lower:
                result = session.run(
                    claim_by_name,
                    user_id=user_id,
                    name=name,
                    name_lower=name_lower,
                    aliases=alias_values,
                    facts=facts,
                    has_new_facts=has_new_facts,
                    facts_updated_ts=facts_updated_ts,
                )
                if result.peek() is not None:
                    return
                conflicting_owner = self._name_lower_owner_user_id(session, name_lower)
            set_name_lower = not conflicting_owner or conflicting_owner == user_id
            if conflicting_owner and conflicting_owner != user_id:
                logger.warning(
                    "Neo4j name_lower conflict while upserting person facts for user_id=%s name_lower=%s; "
                    "owned by user_id=%s. Preserving existing owner.",
                    user_id,
                    name_lower,
                    conflicting_owner,
                )
            session.run(
                merge_by_id,
                user_id=user_id,
                name=name,
                name_lower=name_lower,
                set_name_lower=set_name_lower,
                aliases=alias_values,
                facts=facts,
                has_new_facts=has_new_facts,
                facts_updated_ts=facts_updated_ts,
            )

    def upsert_relationship(self, source_id: str, target_id: str, relationship: str) -> None:
        query = """
        MERGE (a:Person {user_id: $source_id})
        MERGE (b:Person {user_id: $target_id})
        MERGE (a)-[r:RELATES_TO]->(b)
        SET r.kind = $relationship
        """
        with self._session() as session:
            session.run(query, source_id=source_id, target_id=target_id, relationship=relationship)

    def update_last_seen(self, user_id: str, name: str, timestamp: float) -> None:
        if not user_id:
            return
        name = name or ""
        name_lower = name.strip().lower()
        alias_values = [name] if name else []
        last_seen_ts = float(timestamp or 0.0)
        claim_by_name = """
        MATCH (p:Person {name_lower: $name_lower})
        WHERE $name_lower <> "" AND (p.user_id IS NULL OR p.user_id = "")
        SET p.user_id = $user_id,
            p.name = CASE WHEN $name <> "" THEN $name ELSE p.name END,
            p.name_lower = CASE WHEN $name_lower <> "" THEN $name_lower ELSE p.name_lower END,
            p.aliases = reduce(acc = [], x IN coalesce(p.aliases, []) + $aliases |
                CASE WHEN x IS NULL OR trim(x) = "" OR x IN acc THEN acc ELSE acc + x END),
            p.last_seen_ts = CASE
                WHEN coalesce(p.last_seen_ts, 0.0) < $last_seen_ts THEN $last_seen_ts
                ELSE coalesce(p.last_seen_ts, 0.0)
            END
        RETURN p
        """
        query = """
        MERGE (p:Person {user_id: $user_id})
        SET p.name = CASE WHEN $name <> "" THEN $name ELSE p.name END,
            p.name_lower = CASE
                WHEN $set_name_lower AND $name_lower <> "" THEN $name_lower
                ELSE p.name_lower
            END,
            p.aliases = reduce(acc = [], x IN coalesce(p.aliases, []) + $aliases |
                CASE WHEN x IS NULL OR trim(x) = "" OR x IN acc THEN acc ELSE acc + x END),
            p.last_seen_ts = CASE
                WHEN coalesce(p.last_seen_ts, 0.0) < $last_seen_ts THEN $last_seen_ts
                ELSE coalesce(p.last_seen_ts, 0.0)
            END
        """
        with self._session() as session:
            conflicting_owner = ""
            if name_lower:
                result = session.run(
                    claim_by_name,
                    user_id=user_id,
                    name=name,
                    name_lower=name_lower,
                    aliases=alias_values,
                    last_seen_ts=last_seen_ts,
                )
                if result.peek() is not None:
                    return
                conflicting_owner = self._name_lower_owner_user_id(session, name_lower)
            set_name_lower = not conflicting_owner or conflicting_owner == user_id
            if conflicting_owner and conflicting_owner != user_id:
                logger.warning(
                    "Neo4j name_lower conflict while updating last_seen for user_id=%s name_lower=%s; "
                    "owned by user_id=%s. Preserving existing owner.",
                    user_id,
                    name_lower,
                    conflicting_owner,
                )
            session.run(
                query,
                user_id=user_id,
                name=name,
                name_lower=name_lower,
                set_name_lower=set_name_lower,
                aliases=alias_values,
                last_seen_ts=last_seen_ts,
            )

    def init_runtime_marker(self, runtime_key: str, initialized_ts: float) -> None:
        with self._session() as session:
            session.run(
                "MERGE (m:SystemRuntime {name: $name}) "
                "SET m.status = coalesce(m.status, 'idle'), "
                "    m.initialized_ts = coalesce(m.initialized_ts, $initialized_ts)",
                name=runtime_key,
                initialized_ts=float(initialized_ts),
            )

    def mark_runtime_running(self, runtime_key: str, started_ts: float) -> None:
        with self._session() as session:
            session.run(
                "MERGE (m:SystemRuntime {name: $name}) "
                "SET m.status = 'running', "
                "    m.last_started_ts = $last_started_ts, "
                "    m.last_error = ''",
                name=runtime_key,
                last_started_ts=float(started_ts),
            )

    def mark_runtime_success(self, runtime_key: str, payload: dict[str, Any]) -> None:
        data = dict(payload or {})
        status = str(data.pop("status", "success") or "success")
        with self._session() as session:
            session.run(
                "MERGE (m:SystemRuntime {name: $name}) "
                "SET m.status = $status, "
                "    m += $payload",
                name=runtime_key,
                status=status,
                payload=data,
            )

    def mark_runtime_error(self, runtime_key: str, payload: dict[str, Any]) -> None:
        data = dict(payload or {})
        with self._session() as session:
            session.run(
                "MERGE (m:SystemRuntime {name: $name}) "
                "SET m.status = 'error', "
                "    m += $payload",
                name=runtime_key,
                payload=data,
            )

    def fetch_facts_dedupe_candidates(self) -> list[dict[str, Any]]:
        query = (
            "MATCH (p:Person) "
            "WHERE size(coalesce(p.facts, [])) > 1 AND coalesce(p.needs_dedupe, false) = true "
            "RETURN p.user_id as user_id, p.name as name, p.facts as facts"
        )
        with self._session() as session:
            return [dict(record) for record in session.run(query)]

    def apply_deduped_facts(
        self,
        user_id: str,
        refined_facts: list[str],
        *,
        deduped_ts: float,
    ) -> None:
        with self._session() as session:
            session.run(
                "MATCH (p:Person {user_id: $user_id}) "
                "SET p.facts = $facts, "
                "    p.needs_dedupe = false, "
                "    p.facts_deduped_ts = $facts_deduped_ts",
                user_id=user_id,
                facts=refined_facts,
                facts_deduped_ts=float(deduped_ts),
            )

    def mark_facts_deduped(self, user_id: str, *, deduped_ts: float) -> None:
        with self._session() as session:
            session.run(
                "MATCH (p:Person {user_id: $user_id}) "
                "SET p.needs_dedupe = false, "
                "    p.facts_deduped_ts = $facts_deduped_ts",
                user_id=user_id,
                facts_deduped_ts=float(deduped_ts),
            )

    def fetch_experience_dedupe_candidates(
        self,
        *,
        index_name: str,
        neighbor_k: int,
        score_floor: float,
    ) -> list[dict[str, Any]]:
        query = (
            "MATCH (e:Experience) "
            "WHERE e.embedding IS NOT NULL AND e.id IS NOT NULL AND coalesce(e.persona_id, '') <> '' "
            "CALL db.index.vector.queryNodes($index_name, $neighbor_k, e.embedding) YIELD node, score "
            "WHERE node:Experience "
            "  AND node.id IS NOT NULL "
            "  AND e.id < node.id "
            "  AND coalesce(node.persona_id, '') = coalesce(e.persona_id, '') "
            "  AND score >= $score_floor "
            "RETURN "
            "  e.id AS left_id, "
            "  coalesce(e.timestamp, 0.0) AS left_timestamp, "
            "  coalesce(e.timestamp_start, '') AS left_timestamp_start, "
            "  coalesce(e.timestamp_end, '') AS left_timestamp_end, "
            "  node.id AS right_id, "
            "  coalesce(node.timestamp, 0.0) AS right_timestamp, "
            "  coalesce(node.timestamp_start, '') AS right_timestamp_start, "
            "  coalesce(node.timestamp_end, '') AS right_timestamp_end, "
            "  score "
            "ORDER BY score DESC"
        )
        with self._session() as session:
            return [
                dict(record)
                for record in session.run(
                    query,
                    index_name=index_name,
                    neighbor_k=int(neighbor_k),
                    score_floor=float(score_floor),
                )
            ]

    def merge_experience_group(
        self,
        *,
        keep_id: str,
        dup_ids: list[str],
        merged_timestamp: float,
        merged_timestamp_start: str,
        merged_timestamp_end: str,
        deduped_ts: float,
    ) -> int:
        with self._session() as session:
            row = session.run(
                "MATCH (keep:Experience {id: $keep_id}) "
                "SET keep.timestamp = $timestamp, "
                "    keep.timestamp_start = $timestamp_start, "
                "    keep.timestamp_end = $timestamp_end, "
                "    keep.deduped_ts = $deduped_ts, "
                "    keep.dedupe_count = toInteger(coalesce(keep.dedupe_count, 1)) + $duplicate_count, "
                "    keep.merged_experience_ids = reduce(acc = coalesce(keep.merged_experience_ids, []), x IN $dup_ids | "
                "        CASE WHEN x IN acc THEN acc ELSE acc + x END) "
                "WITH keep "
                "MATCH (dup:Experience) "
                "WHERE dup.id IN $dup_ids "
                "DETACH DELETE dup "
                "RETURN count(dup) AS deleted_count",
                keep_id=keep_id,
                timestamp=float(merged_timestamp),
                timestamp_start=merged_timestamp_start,
                timestamp_end=merged_timestamp_end,
                deduped_ts=float(deduped_ts),
                duplicate_count=len(dup_ids),
                dup_ids=dup_ids,
            ).single()
        return int(row.get("deleted_count", 0) if row else 0)

    @property
    def driver(self):
        return self._driver

    @property
    def database(self) -> str:
        return self._database
