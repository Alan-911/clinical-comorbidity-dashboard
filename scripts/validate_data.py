"""Validate checked-in clinical mining data without modifying it."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(relative_path: str, required_columns: set[str]) -> list[dict[str, str]]:
    path = ROOT / relative_path
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{relative_path}: missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{relative_path}: no rows")
    return rows


def main() -> None:
    events = read_rows("data_raw/raw_data.csv",
                       {"PatientID", "VisitID", "AgeGroup", "MedicalEvent"})
    transactions = read_rows("transactions/transactions.csv", {"VisitID", "Items"})
    rules = read_rows("data_processed/association_rules.csv",
                     {"antecedents", "consequents", "support", "confidence", "lift"})

    visit_ids = [row["VisitID"] for row in transactions]
    if len(visit_ids) != len(set(visit_ids)):
        raise ValueError("Multiple transaction rows for the same VisitID")
    event_visit_ids = {row["VisitID"] for row in events}
    if not set(visit_ids).issubset(event_visit_ids):
        raise ValueError("Some transactions have no corresponding raw events")

    for index, rule in enumerate(rules, start=1):
        for column in ("support", "confidence", "lift"):
            value = float(rule[column])
            if value != value or value < 0:
                raise ValueError(f"Rule {index}: invalid {column}")
            if column in ("support", "confidence") and value > 1:
                raise ValueError(f"Rule {index}: {column} must be <= 1")

    print(f"Raw events: {len(events)}")
    print(f"Visit transactions: {len(transactions)}")
    print(f"Cached association rules: {len(rules)}")
    print("Data schema and basic integrity checks passed.")


if __name__ == "__main__":
    main()
