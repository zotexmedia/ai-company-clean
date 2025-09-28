"""Export canonical mappings to CSV for downstream systems."""

from __future__ import annotations

import argparse
import csv

from sqlalchemy import select

from app.stores.db import Alias, CanonicalCompany, SessionLocal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export alias->canonical mappings")
    parser.add_argument("output", help="Destination CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with SessionLocal() as session:
        stmt = (
            select(
                Alias.alias_name,
                CanonicalCompany.canonical_name,
                Alias.confidence_last,
                Alias.source,
                CanonicalCompany.key_form,
            )
            .join(CanonicalCompany, Alias.canonical_id == CanonicalCompany.id)
        )
        rows = session.execute(stmt).all()

    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alias_name", "canonical_name", "confidence", "source", "key_form"])
        for row in rows:
            writer.writerow(row)
    print(f"Exported {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
