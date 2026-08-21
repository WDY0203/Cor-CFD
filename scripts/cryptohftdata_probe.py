#!/usr/bin/env python3
"""Inspect CryptoHFTData hourly order-book objects without loading them fully into RAM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq
import zstandard as zstd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decompress_zstd(source: Path, destination: Path) -> None:
    dctx = zstd.ZstdDecompressor()
    with source.open("rb") as src, destination.open("wb") as dst:
        with dctx.stream_reader(src) as reader:
            shutil.copyfileobj(reader, dst, length=8 * 1024 * 1024)


def infer_time_unit(value: int | float | None) -> tuple[str | None, float | None]:
    if value is None:
        return None, None
    magnitude = abs(float(value))
    if magnitude >= 1e17:
        return "ns", 1_000_000.0
    if magnitude >= 1e14:
        return "us", 1_000.0
    if magnitude >= 1e11:
        return "ms", 1.0
    if magnitude >= 1e8:
        return "s", 0.001
    return "unknown", None


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def query_records(connection: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, map(json_safe, row))) for row in cursor.fetchall()]


def quoted_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quoted_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def inspect_file(source: Path, work_dir: Path) -> dict[str, Any]:
    parquet_path = work_dir / source.name.removesuffix(".zst")
    decompress_zstd(source, parquet_path)

    parquet = pq.ParquetFile(parquet_path)
    metadata = parquet.metadata
    schema = parquet.schema_arrow
    columns = schema.names

    connection = duckdb.connect(database=":memory:")
    connection.execute("PRAGMA threads=4")
    connection.execute("PRAGMA memory_limit='5GB'")
    connection.execute(
        f"CREATE VIEW p AS SELECT * FROM read_parquet({quoted_literal(str(parquet_path))})"
    )

    report: dict[str, Any] = {
        "source_file": source.name,
        "compressed_bytes": source.stat().st_size,
        "compressed_sha256": sha256_file(source),
        "parquet_bytes": parquet_path.stat().st_size,
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "created_by": metadata.created_by,
        "columns": [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in schema],
    }

    if "event_type" in columns:
        report["event_type_rows"] = query_records(
            connection,
            "SELECT CAST(event_type AS VARCHAR) AS event_type, count(*) AS rows "
            "FROM p GROUP BY 1 ORDER BY rows DESC",
        )
    if "side" in columns:
        report["side_rows"] = query_records(
            connection,
            "SELECT lower(CAST(side AS VARCHAR)) AS side, count(*) AS rows "
            "FROM p GROUP BY 1 ORDER BY rows DESC",
        )

    time_candidates = [
        name
        for name in ("received_time", "event_time", "transaction_time", "trade_time", "cts", "ts")
        if name in columns
    ]
    report["time_columns"] = {}
    for name in time_candidates:
        qname = quoted_identifier(name)
        row = query_records(
            connection,
            f"SELECT count({qname}) AS non_null, count(DISTINCT {qname}) AS distinct_values, "
            f"min({qname}) AS min_value, max({qname}) AS max_value FROM p",
        )[0]
        unit, units_per_ms = infer_time_unit(row.get("min_value"))
        row["inferred_unit"] = unit
        row["units_per_ms"] = units_per_ms
        report["time_columns"][name] = row

    id_candidates = [
        name
        for name in (
            "first_update_id",
            "final_update_id",
            "prev_final_update_id",
            "last_update_id",
            "update_id",
            "sequence",
            "seq",
            "u",
            "pu",
        )
        if name in columns
    ]
    report["update_id_columns"] = {}
    for name in id_candidates:
        qname = quoted_identifier(name)
        report["update_id_columns"][name] = query_records(
            connection,
            f"SELECT count({qname}) AS non_null, count(DISTINCT {qname}) AS distinct_values, "
            f"min({qname}) AS min_value, max({qname}) AS max_value FROM p",
        )[0]

    key_columns: list[str] = []
    if "received_time" in columns:
        key_columns.append("received_time")
    elif "event_time" in columns:
        key_columns.append("event_time")
    if "event_type" in columns:
        key_columns.append("event_type")
    key_columns.extend(id_candidates)
    # Preserve order while removing duplicates.
    key_columns = list(dict.fromkeys(key_columns))
    report["message_key_columns"] = key_columns

    if key_columns:
        key_sql = ", ".join(quoted_identifier(name) for name in key_columns)
        event_type_expr = (
            "lower(CAST(event_type AS VARCHAR))" if "event_type" in columns else "NULL::VARCHAR"
        )
        bid_expr = (
            "count(*) FILTER (WHERE lower(CAST(side AS VARCHAR)) = 'bid')"
            if "side" in columns
            else "0"
        )
        ask_expr = (
            "count(*) FILTER (WHERE lower(CAST(side AS VARCHAR)) = 'ask')"
            if "side" in columns
            else "0"
        )
        connection.execute(
            f"CREATE TEMP TABLE messages AS "
            f"SELECT {key_sql}, {event_type_expr} AS _event_type, count(*) AS rows_per_message, "
            f"{bid_expr} AS bid_rows, {ask_expr} AS ask_rows "
            f"FROM p GROUP BY {key_sql}"
        )
        report["message_summary"] = query_records(
            connection,
            "SELECT count(*) AS messages, avg(rows_per_message) AS mean_rows, "
            "min(rows_per_message) AS min_rows, "
            "quantile_cont(rows_per_message, 0.50) AS p50_rows, "
            "quantile_cont(rows_per_message, 0.90) AS p90_rows, "
            "quantile_cont(rows_per_message, 0.99) AS p99_rows, "
            "max(rows_per_message) AS max_rows FROM messages",
        )[0]
        report["message_types"] = query_records(
            connection,
            "SELECT _event_type AS event_type, count(*) AS messages, sum(rows_per_message) AS rows, "
            "avg(rows_per_message) AS mean_rows, quantile_cont(rows_per_message, 0.50) AS p50_rows, "
            "quantile_cont(rows_per_message, 0.90) AS p90_rows, "
            "quantile_cont(rows_per_message, 0.99) AS p99_rows, max(rows_per_message) AS max_rows "
            "FROM messages GROUP BY 1 ORDER BY messages DESC",
        )
        report["largest_messages"] = query_records(
            connection,
            "SELECT _event_type AS event_type, rows_per_message, bid_rows, ask_rows "
            "FROM messages ORDER BY rows_per_message DESC LIMIT 20",
        )
        if "event_type" in columns:
            report["snapshot_message_depth"] = query_records(
                connection,
                "SELECT count(*) AS snapshots, avg(rows_per_message) AS mean_rows, "
                "min(rows_per_message) AS min_rows, "
                "quantile_cont(rows_per_message, 0.50) AS p50_rows, "
                "quantile_cont(rows_per_message, 0.90) AS p90_rows, "
                "quantile_cont(rows_per_message, 0.99) AS p99_rows, "
                "max(rows_per_message) AS max_rows, "
                "quantile_cont(bid_rows, 0.50) AS p50_bid_rows, "
                "quantile_cont(ask_rows, 0.50) AS p50_ask_rows "
                "FROM messages WHERE _event_type = 'snapshot'",
            )[0]

        cadence_column = "received_time" if "received_time" in key_columns else "event_time"
        cadence_meta = report["time_columns"].get(cadence_column, {})
        units_per_ms = cadence_meta.get("units_per_ms")
        qcad = quoted_identifier(cadence_column)
        connection.execute(
            f"CREATE TEMP TABLE cadence AS "
            f"SELECT {qcad} AS ts, {qcad} - lag({qcad}) OVER (ORDER BY {qcad}) AS delta_raw "
            f"FROM messages"
        )
        cadence: dict[str, Any] = {
            "column": cadence_column,
            "inferred_unit": cadence_meta.get("inferred_unit"),
            "units_per_ms": units_per_ms,
        }
        raw_stats = query_records(
            connection,
            "SELECT count(delta_raw) AS intervals, "
            "min(delta_raw) AS min_raw, "
            "quantile_cont(delta_raw, 0.01) AS p01_raw, "
            "quantile_cont(delta_raw, 0.10) AS p10_raw, "
            "quantile_cont(delta_raw, 0.50) AS p50_raw, "
            "quantile_cont(delta_raw, 0.90) AS p90_raw, "
            "quantile_cont(delta_raw, 0.99) AS p99_raw, "
            "max(delta_raw) AS max_raw, "
            "count(*) FILTER (WHERE delta_raw = 0) AS zero_intervals "
            "FROM cadence WHERE delta_raw IS NOT NULL",
        )[0]
        cadence["raw_stats"] = raw_stats
        if units_per_ms:
            ms_stats = {}
            for key, value in raw_stats.items():
                if key.endswith("_raw") and value is not None:
                    ms_stats[key.removesuffix("_raw") + "_ms"] = float(value) / float(units_per_ms)
                elif key in ("intervals", "zero_intervals"):
                    ms_stats[key] = value
            cadence["ms_stats"] = ms_stats
            thresholds = [1, 5, 10, 20, 50, 100, 250, 500, 1000]
            threshold_expr = ", ".join(
                f"avg(CASE WHEN delta_raw <= {threshold * units_per_ms:.12g} THEN 1.0 ELSE 0.0 END) "
                f"AS frac_le_{threshold}ms"
                for threshold in thresholds
            )
            cadence["threshold_fractions"] = query_records(
                connection,
                f"SELECT {threshold_expr} FROM cadence WHERE delta_raw IS NOT NULL",
            )[0]
            cadence["top_rounded_ms_intervals"] = query_records(
                connection,
                f"SELECT round(delta_raw / {units_per_ms:.12g}, 1) AS rounded_ms, count(*) AS count "
                "FROM cadence WHERE delta_raw IS NOT NULL AND delta_raw >= 0 "
                "GROUP BY 1 ORDER BY count DESC, rounded_ms LIMIT 30",
            )
        report["message_cadence"] = cadence

    order_column = "received_time" if "received_time" in columns else time_candidates[0] if time_candidates else None
    select_columns = [
        name
        for name in (
            "received_time",
            "event_time",
            "transaction_time",
            "event_type",
            "first_update_id",
            "final_update_id",
            "prev_final_update_id",
            "last_update_id",
            "side",
            "price",
            "quantity",
            "order_count",
        )
        if name in columns
    ]
    if select_columns:
        select_sql = ", ".join(quoted_identifier(name) for name in select_columns)
        order_sql = f" ORDER BY {quoted_identifier(order_column)}" if order_column else ""
        report["first_rows"] = query_records(connection, f"SELECT {select_sql} FROM p{order_sql} LIMIT 30")
        if order_column:
            report["last_rows"] = query_records(
                connection,
                f"SELECT {select_sql} FROM p ORDER BY {quoted_identifier(order_column)} DESC LIMIT 30",
            )

    connection.close()
    parquet.close()
    parquet_path.unlink(missing_ok=True)
    return report


def render_markdown(reports: list[dict[str, Any]]) -> str:
    lines = ["# CryptoHFTData probe", ""]
    for report in reports:
        lines.extend(
            [
                f"## {report['source_file']}",
                "",
                f"- Compressed bytes: `{report['compressed_bytes']:,}`",
                f"- SHA256: `{report['compressed_sha256']}`",
                f"- Parquet bytes: `{report['parquet_bytes']:,}`",
                f"- Rows: `{report['rows']:,}`",
                f"- Row groups: `{report['row_groups']}`",
                f"- Columns: `{', '.join(item['name'] for item in report['columns'])}`",
                "",
            ]
        )
        if report.get("event_type_rows"):
            lines.append("### Event types")
            lines.append("")
            for item in report["event_type_rows"]:
                lines.append(f"- `{item['event_type']}`: `{item['rows']:,}` rows")
            lines.append("")
        if report.get("message_summary"):
            summary = report["message_summary"]
            lines.extend(
                [
                    "### Reconstructed message groups",
                    "",
                    f"- Messages: `{summary['messages']:,}`",
                    f"- Mean rows/message: `{float(summary['mean_rows']):.3f}`",
                    f"- P50/P90/P99/max rows: `{summary['p50_rows']}` / `{summary['p90_rows']}` / `{summary['p99_rows']}` / `{summary['max_rows']}`",
                    "",
                ]
            )
        cadence = report.get("message_cadence", {}).get("ms_stats")
        if cadence:
            lines.extend(
                [
                    "### Message cadence based on receive time",
                    "",
                    f"- Intervals: `{cadence.get('intervals', 0):,}`",
                    f"- P01/P10/P50/P90/P99: `{cadence.get('p01_ms')}` / `{cadence.get('p10_ms')}` / `{cadence.get('p50_ms')}` / `{cadence.get('p90_ms')}` / `{cadence.get('p99_ms')}` ms",
                    f"- Min/max: `{cadence.get('min_ms')}` / `{cadence.get('max_ms')}` ms",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.raw_dir.glob("*.parquet.zst"))
    if not sources:
        raise SystemExit(f"no .parquet.zst files under {args.raw_dir}")

    with tempfile.TemporaryDirectory(prefix="chd-probe-") as temporary:
        work_dir = Path(temporary)
        reports = [inspect_file(source, work_dir) for source in sources]

    payload = {"files": reports}
    (args.out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_safe) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "summary.md").write_text(render_markdown(reports), encoding="utf-8")
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
