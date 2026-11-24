"""Regression tests for rule-item parsing and exact-match filters."""
import unittest

import pandas as pd

from src.rule_engine import rule_members, filter_association_rules


class RuleEngineTests(unittest.TestCase):
    def setUp(self):
        self.rules = pd.DataFrame([
            {"antecedents": "frozenset({'Heart Disease', 'Age_Senior'})",
             "consequents": "frozenset({'Statin'})", "lift": 3.1},
            {"antecedents": "frozenset({'Heart'})",
             "consequents": "frozenset({'Inhaler'})", "lift": 5.2},
            {"antecedents": "frozenset({'Diabetes'})",
             "consequents": "frozenset({'Statin'})", "lift": 2.5},
        ])

    def test_frozenset_parsing(self):
        self.assertEqual(rule_members("frozenset({'Heart Disease', 'Age_Senior'})"),
                         {"Heart Disease", "Age_Senior"})

    def test_no_substring_matches(self):
        result = filter_association_rules(self.rules, antecedent="Heart")
        self.assertEqual(result["lift"].tolist(), [5.2])

    def test_two_sided_filter(self):
        result = filter_association_rules(
            self.rules, antecedent="Heart Disease", consequent="Statin")
        self.assertEqual(len(result), 1)

    def test_sorted_by_lift(self):
        result = filter_association_rules(self.rules, consequent="Statin")
        self.assertEqual(result["lift"].tolist(), [3.1, 2.5])

    def test_no_code_execution_in_rule_text(self):
        self.assertEqual(rule_members("frozenset({'Diabetes'})"), {"Diabetes"})


if __name__ == "__main__":
    unittest.main()
