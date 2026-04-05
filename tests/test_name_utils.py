import pytest
from gpt5_roleplay_system.name_utils import (
    split_display_and_username,
    extract_username,
    extract_display_name,
    normalize_display_name,
    normalize_for_match,
    name_matches,
)

@pytest.mark.parametrize("full_name, expected", [
    ("Display Name (username)", ("Display Name", "username")),
    ("First Last (Username)", ("First Last", "Username")),
    ("username", ("username", "username")),
    ("", ("", "")),
    ("   ", ("", "")),
    ("Only Display", ("Only Display", "Only Display")),
    ("(username)", ("username", "username")),
    ("  Display   Name  (username)  ", ("Display Name", "username")),
    ("Display (Extra) (username)", ("Display (Extra)", "username")),
    ("Display (username) ", ("Display", "username")),
])
def test_split_display_and_username(full_name, expected):
    assert split_display_and_username(full_name) == expected

def test_extract_username():
    assert extract_username("Display (user)") == "user"
    assert extract_username("user") == "user"
    assert extract_username("") == ""

def test_extract_display_name():
    assert extract_display_name("Display (user)") == "Display"
    assert extract_display_name("user") == "user"
    assert extract_display_name("") == ""

@pytest.mark.parametrize("name, expected", [
    ("Display (user)", "display"),
    ("  Mixed   CASE  ", "mixed case"),
    ("", ""),
    (None, ""),
    ("Name (handle) extra", "name"),
])
def test_normalize_display_name(name, expected):
    assert normalize_display_name(name) == expected

@pytest.mark.parametrize("name, expected", [
    ("Café (user)", "cafe"),
    ("Display-Name! 123", "displayname123"),
    ("", ""),
    ("   ", ""),
])
def test_normalize_for_match(name, expected):
    assert normalize_for_match(name) == expected

@pytest.mark.parametrize("sender, candidate, expected", [
    ("User (handle)", "User", True),
    ("User", "User (handle)", True),
    ("John Doe", "John", True),
    ("John", "John Doe", True),
    ("Café", "Cafe", True),
    ("Different", "Names", False),
    ("", "User", False),
    ("User", "", False),
    ("User123", "User", True), # normalize_for_match prefix match
])
def test_name_matches(sender, candidate, expected):
    assert name_matches(sender, candidate) == expected
