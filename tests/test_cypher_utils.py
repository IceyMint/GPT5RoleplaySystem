
import unittest
import sys
import os

# Add src to sys.path
sys.path.append(os.path.abspath("src"))

from gpt5_roleplay_system.cypher_utils import quote_cypher_identifier

class TestCypherUtils(unittest.TestCase):
    def test_quote_cypher_identifier_supports_hyphenated_database_names(self):
        self.assertEqual(quote_cypher_identifier("gpt5-roleplay"), "`gpt5-roleplay`")

    def test_quote_cypher_identifier_escapes_backticks(self):
        self.assertEqual(quote_cypher_identifier("db`name"), "`db``name`")

    def test_quote_cypher_identifier_empty(self):
        self.assertEqual(quote_cypher_identifier(""), "`neo4j`")
        self.assertEqual(quote_cypher_identifier(None), "`neo4j`")

if __name__ == "__main__":
    unittest.main()
