from __future__ import annotations

import re
import unicodedata

_USERNAME_PAREN_RE = re.compile(r"\(([^()]+)\)\s*$")


def split_display_and_username(full_name: str) -> tuple[str, str]:
    """Split a viewer full-name into (display_name, username).

    Supported shapes:
    - "Displayname (username)"
    - "First Last (Username)"
    - "username"
    """
    if not full_name:
        return "", ""
    cleaned = " ".join(str(full_name).strip().split())
    match = _USERNAME_PAREN_RE.search(cleaned)
    if not match:
        return cleaned, cleaned
    username = match.group(1).strip()
    display = cleaned[: match.start()].strip()
    if not display:
        display = username
    return display, username


def extract_username(full_name: str) -> str:
    """Return the username portion from a viewer full-name string."""
    return split_display_and_username(full_name)[1]


def extract_display_name(full_name: str) -> str:
    """Return the display-name portion from a viewer full-name string."""
    return split_display_and_username(full_name)[0]


def normalize_display_name(name: str) -> str:
    """Normalize a name for display-oriented comparisons.

    This strips parenthetical account handles, collapses whitespace, and
    case-folds the result, while preserving most punctuation.
    """
    if not name:
        return ""
    base = name.split("(", 1)[0].strip()
    collapsed = " ".join(base.split())
    return collapsed.casefold()


def normalize_for_match(name: str) -> str:
    """Normalize a name aggressively for matching and self-filtering.

    This removes diacritics and non-alphanumerics for stable matching.
    """
    display = normalize_display_name(name)
    if not display:
        return ""
    display = unicodedata.normalize("NFKC", display)
    # Strip diacritics and combining marks, then drop non-alphanumerics.
    display = "".join(ch for ch in unicodedata.normalize("NFKD", display) if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "", display)


def name_matches(sender: str, candidate: str) -> bool:
    """Return True when two names should be considered the same entity."""
    sender_display = normalize_display_name(sender)
    candidate_display = normalize_display_name(candidate)
    if not sender_display or not candidate_display:
        return False

    if sender_display == candidate_display:
        return True
    if sender_display.startswith(candidate_display + " "):
        return True
    if candidate_display.startswith(sender_display + " "):
        return True

    sender_match = normalize_for_match(sender)
    candidate_match = normalize_for_match(candidate)
    if not sender_match or not candidate_match:
        return False
    if sender_match == candidate_match:
        return True
    if sender_match.startswith(candidate_match):
        return True
    if candidate_match.startswith(sender_match):
        return True
    return False
