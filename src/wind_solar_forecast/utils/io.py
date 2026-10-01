"""I/O utilities for Parquet, Zarr, and DuckDB storage and analytics."""

from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from wind_solar_forecast.utils.logging import logger


def save_parquet(
    df: pd.DataFrame,
    path: str | Path,
    partition_cols: list[str] | None = None
) -> Path:
    """Saves DataFrame to Parquet format, creating parent directories if needed.

    Args:
        df: Pandas DataFrame to persist.
        path: Destination path or directory if partitioned.
        partition_cols: Optional columns to partition dataset by.

    Returns:
        Path to written parquet file or directory.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df)
    if partition_cols:
        target.mkdir(parents=True, exist_ok=True)
        pq.write_to_dataset(table, root_path=str(target), partition_cols=partition_cols)
    else:
        pq.write_table(table, str(target), compression="snappy")

    logger.debug("Saved Parquet dataset", path=str(target), rows=len(df))
    return target


def load_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Loads DataFrame from Parquet file or directory.

    Args:
        path: Path to parquet file or partitioned directory.
        columns: Optional list of columns to load.

    Returns:
        Loaded DataFrame.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parquet source does not exist: {p}")

    df = pd.read_parquet(p, columns=columns)
    return df


def get_duckdb_connection(db_path: str | Path = "data/processed/power_market.duckdb") -> duckdb.DuckDBPyConnection:
    """Opens DuckDB connection and sets up memory and thread defaults.

    Args:
        db_path: Path to DuckDB database file.

    Returns:
        DuckDBPyConnection object.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("PRAGMA threads=4;")
    return con


def register_parquet_views(con: duckdb.DuckDBPyConnection, base_dir: str | Path = "data/feature_store") -> None:
    """Registers auto-updating views for all parquet files in directory for SQL querying.

    Args:
        con: DuckDB connection.
        base_dir: Root directory containing feature store or processed parquet files.
    """
    p = Path(base_dir)
    if not p.exists():
        return

    parquet_files = list(p.rglob("*.parquet"))
    if parquet_files:
        con.execute(f"CREATE OR REPLACE VIEW features AS SELECT * FROM read_parquet('{p}/**/*.parquet');")
        logger.info("Registered DuckDB view 'features'", path=str(p))
