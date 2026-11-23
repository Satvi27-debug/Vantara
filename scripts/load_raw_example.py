"""Example: load raw Online Retail II and validate (run from repo root)."""

from pathlib import Path

from src.data.load_data import load_from_excel
from src.data.validate import validate_basic


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    excel_path = repo_root / "data" / "raw" / "online_retail_II.xlsx"
    df = load_from_excel(excel_path)
    summary = validate_basic(df)
    print("Loaded OK:", summary)


if __name__ == "__main__":
    main()
