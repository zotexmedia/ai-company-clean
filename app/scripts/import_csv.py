"""Utility script to enqueue a normalization job from a local CSV."""

from __future__ import annotations

import argparse

from app.workers.normalize_worker import enqueue_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enqueue a normalization job from CSV")
    parser.add_argument("csv_path", help="Path to local CSV with columns id,raw_name,source,country_hint")
    parser.add_argument("--source", default="csv", choices=["csv", "gmaps", "apollo"], help="Input source tag")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job = enqueue_job(args.csv_path, None, args.source)
    print(f"Enqueued job {job.id} (status={job.status.value})")


if __name__ == "__main__":
    main()
