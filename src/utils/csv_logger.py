# src/utils/csv_logger.py
"""
Shared CSV utility used by method1 and method2 loggers.
Removes duplicated csv.DictWriter boilerplate.
"""
import csv
import os


def init_csv(path: str, headers: list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=headers).writeheader()


def append_row(path: str, headers: list, row: dict) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=headers).writerow(row)
