"""Small, independently testable association-rule helpers."""
import re

import pandas as pd


def rule_members(value: object) -> set[str]:
    """Parse mlxtend's serialized frozenset values or comma-separated rule items.

    No eval() is used: repository CSV values are data, not executable code.
    """
    value = str(value).strip()
    if value.startswith("frozenset("):
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", value)
        return {v.strip() for v in quoted if v.strip()}
    return {v.strip() for v in value.split(",") if v.strip()}


def filter_association_rules(
    rules: pd.DataFrame, antecedent: str = "All", consequent: str = "All"
) -> pd.DataFrame:
    """Filter by exact item membership; return highest-lift rules first."""
    result = rules
    if antecedent != "All":
        result = result[result["antecedents"].map(lambda x: antecedent in rule_members(x))]
    if consequent != "All":
        result = result[result["consequents"].map(lambda x: consequent in rule_members(x))]
    return result.sort_values("lift", ascending=False).copy()
