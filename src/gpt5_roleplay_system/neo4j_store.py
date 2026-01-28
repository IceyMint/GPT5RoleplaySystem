from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


@dataclass
class PersonProfile:
    user_id: str
    name: str
    facts: List[str] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    last_seen_ts: float = 0.0


class KnowledgeStore:
    def fetch_people(self, user_ids: List[str]) -> Dict[str, PersonProfile]:
        raise NotImplementedError

    def fetch_people_by_name(self, names: List[str]) -> Dict[str, PersonProfile]:
        raise NotImplementedError

    def fetch_people_by_partial_name(self, names: List[str]) -> Dict[str, PersonProfile]:
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

    def fetch_people(self, user_ids: List[str]) -> Dict[str, PersonProfile]:
        return {user_id: self._people[user_id] for user_id in user_ids if user_id in self._people}

    def fetch_people_by_name(self, names: List[str]) -> Dict[str, PersonProfile]:
        if not names:
            return {}
        names_lower = {name.lower() for name in names if name}
        results: Dict[str, PersonProfile] = {}
        for profile in self._people.values():
            if profile.name.lower() in names_lower:
                results[profile.user_id] = profile
        return results

    def fetch_people_by_partial_name(self, names: List[str]) -> Dict[str, PersonProfile]:
        if not names:
            return {}
        tokens = [name.lower() for name in names if name]
        results: Dict[str, PersonProfile] = {}
        for profile in self._people.values():
            name_lower = profile.name.lower()
            if any(token in name_lower for token in tokens):
                results[profile.user_id] = profile
        return results

    def upsert_person_facts(self, user_id: str, name: str, facts: List[str]) -> None:
        profile = self._people.get(user_id) or PersonProfile(user_id=user_id, name=name)
        profile.name = name or profile.name
        for fact in facts:
            if fact and fact not in profile.facts:
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
        # Try to create the target database first; ignore if not permitted.
        try:
            with self._system_session() as system_session:
                system_session.run(f"CREATE DATABASE {self._database} IF NOT EXISTS")
        except Exception:
            pass

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
                except Exception:
                    # Ignore if unsupported syntax or constraint cannot be created.
                    continue

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
                    facts=_dedupe_preserve_order(person.get("facts", []) or []),
                    relationships=rels or [],
                    last_seen_ts=float(person.get("last_seen_ts", 0.0) or 0.0),
                )
                profiles[profile.user_id] = profile
        return profiles

    def upsert_person_facts(self, user_id: str, name: str, facts: List[str]) -> None:
        name = name or ""
        name_lower = name.strip().lower()
        facts = _dedupe_preserve_order([fact for fact in facts if fact])
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
                CASE WHEN x IS NULL OR trim(x) = "" OR x IN acc THEN acc ELSE acc + x END)
        RETURN p
        """
        merge_by_id = """
        MERGE (p:Person {user_id: $user_id})
        SET p.name = CASE WHEN $name <> "" THEN $name ELSE p.name END,
            p.name_lower = CASE WHEN $name_lower <> "" THEN $name_lower ELSE p.name_lower END,
            p.aliases = reduce(acc = [], x IN coalesce(p.aliases, []) + $aliases |
                CASE WHEN x IS NULL OR trim(x) = "" OR x IN acc THEN acc ELSE acc + x END),
            p.facts = reduce(acc = [], x IN coalesce(p.facts, []) + $facts |
                CASE WHEN x IS NULL OR trim(x) = "" OR x IN acc THEN acc ELSE acc + x END)
        """
        with self._session() as session:
            if user_id and name_lower:
                result = session.run(
                    claim_by_name,
                    user_id=user_id,
                    name=name,
                    name_lower=name_lower,
                    aliases=alias_values,
                    facts=facts,
                )
                if result.peek() is not None:
                    return
            session.run(
                merge_by_id,
                user_id=user_id,
                name=name,
                name_lower=name_lower,
                aliases=alias_values,
                facts=facts,
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
        query = """
        MERGE (p:Person {user_id: $user_id})
        SET p.name = CASE WHEN $name <> "" THEN $name ELSE p.name END,
            p.name_lower = CASE WHEN $name_lower <> "" THEN $name_lower ELSE p.name_lower END,
            p.aliases = reduce(acc = [], x IN coalesce(p.aliases, []) + $aliases |
                CASE WHEN x IS NULL OR trim(x) = "" OR x IN acc THEN acc ELSE acc + x END),
            p.last_seen_ts = CASE
                WHEN coalesce(p.last_seen_ts, 0.0) < $last_seen_ts THEN $last_seen_ts
                ELSE coalesce(p.last_seen_ts, 0.0)
            END
        """
        with self._session() as session:
            session.run(
                query,
                user_id=user_id,
                name=name,
                name_lower=name_lower,
                aliases=alias_values,
                last_seen_ts=last_seen_ts,
            )

    @property
    def driver(self):
        return self._driver

    @property
    def database(self) -> str:
        return self._database
