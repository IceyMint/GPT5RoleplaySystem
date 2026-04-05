from __future__ import annotations

def quote_cypher_identifier(identifier: str) -> str:
    """Quote a Cypher identifier with backticks, escaping existing backticks."""
    value = str(identifier or "").strip() or "neo4j"
    # Cypher uses doubled backticks to escape backticks within identifiers.
    escaped = value.replace("`", "``")
    return f"`{escaped}`"
