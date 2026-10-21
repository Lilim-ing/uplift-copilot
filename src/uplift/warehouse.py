"""Read-only warehouse over local parquet fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import duckdb
import pandas as pd

TABLES = ("members", "products", "orders", "order_items", "behaviors", "campaigns", "campaign_assignments")


class Warehouse:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        missing = [t for t in TABLES if not (self.data_dir / f"{t}.parquet").exists()]
        if missing:
            raise FileNotFoundError(f"missing fixtures in {self.data_dir}: {missing}")
        self._con = duckdb.connect(database=":memory:", config={"memory_limit": "512MB"})
        for table in TABLES:
            path = (self.data_dir / f"{table}.parquet").as_posix()
            self._con.execute(f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{path}')")

    def query(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        return self._con.execute(sql, list(params or [])).fetchdf()

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> Warehouse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
