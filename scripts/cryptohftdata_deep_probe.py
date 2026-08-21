#!/usr/bin/env python3
"""Targeted second-pass analysis for the two CryptoHFTData sample objects."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import zstandard as zstd


def decompress(source: Path, destination: Path) -> None:
    decoder = zstd.ZstdDecompressor()
    with source.open("rb") as src, destination.open("wb") as dst:
        with decoder.stream_reader(src) as reader:
            shutil.copyfileobj(reader, dst, 8 * 1024 * 1024)


def safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def records(conn: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    cur = conn.execute(sql)
    names = [column[0] for column in cur.description]
    return [dict(zip(names, (safe(value) for value in row))) for row in cur.fetchall()]


def scalar(conn: duckdb.DuckDBPyConnection, sql: str) -> Any:
    return safe(conn.execute(sql).fetchone()[0])


def make_connection(parquet_path: Path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(database=":memory:")
    conn.execute("PRAGMA threads=4")
    conn.execute("PRAGMA memory_limit='5GB'")
    escaped = str(parquet_path).replace("'", "''")
    conn.execute(f"CREATE VIEW p AS SELECT * FROM read_parquet('{escaped}')")
    return conn


def cadence_sql(table: str, partition: str | None = None) -> str:
    partition_sql = f"PARTITION BY {partition} " if partition else ""
    grouping = f", {partition}" if partition else ""
    select_group = f"{partition}, " if partition else ""
    group_by = f" GROUP BY {partition} ORDER BY {partition}" if partition else ""
    return f"""
        WITH ordered AS (
            SELECT * EXCLUDE (received_time, event_time, transaction_time),
                   received_time,
                   event_time,
                   transaction_time,
                   (received_time - lag(received_time) OVER ({partition_sql}ORDER BY received_time)) / 1000000.0 AS receive_gap_ms,
                   event_time - lag(event_time) OVER ({partition_sql}ORDER BY received_time) AS event_gap_ms,
                   transaction_time - lag(transaction_time) OVER ({partition_sql}ORDER BY received_time) AS transaction_gap_ms
            FROM {table}
        )
        SELECT {select_group}
               count(receive_gap_ms) AS intervals,
               quantile_cont(receive_gap_ms, 0.01) AS receive_p01_ms,
               quantile_cont(receive_gap_ms, 0.10) AS receive_p10_ms,
               quantile_cont(receive_gap_ms, 0.50) AS receive_p50_ms,
               quantile_cont(receive_gap_ms, 0.90) AS receive_p90_ms,
               quantile_cont(receive_gap_ms, 0.99) AS receive_p99_ms,
               max(receive_gap_ms) AS receive_max_ms,
               quantile_cont(event_gap_ms, 0.50) AS event_p50_ms,
               quantile_cont(event_gap_ms, 0.90) AS event_p90_ms,
               quantile_cont(event_gap_ms, 0.99) AS event_p99_ms,
               quantile_cont(transaction_gap_ms, 0.50) AS transaction_p50_ms,
               quantile_cont(transaction_gap_ms, 0.90) AS transaction_p90_ms,
               quantile_cont(transaction_gap_ms, 0.99) AS transaction_p99_ms
        FROM ordered
        WHERE receive_gap_ms IS NOT NULL{group_by}
    """


def analyze_bybit(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    conn.execute("""
        CREATE TEMP TABLE messages AS
        SELECT received_time, event_time, transaction_time,
               final_update_id, last_update_id,
               count(*) AS rows_per_message,
               count(*) FILTER (WHERE lower(CAST(side AS VARCHAR)) = 'bid') AS bid_rows,
               count(*) FILTER (WHERE lower(CAST(side AS VARCHAR)) = 'ask') AS ask_rows
        FROM p
        GROUP BY received_time, event_time, transaction_time, final_update_id, last_update_id
    """)

    unique_ids = [row[0] for row in conn.execute(
        "SELECT DISTINCT final_update_id FROM messages WHERE final_update_id IS NOT NULL ORDER BY 1"
    ).fetchall()]
    gaps = sorted(
        ((unique_ids[index] - unique_ids[index - 1], unique_ids[index - 1], unique_ids[index])
         for index in range(1, len(unique_ids))),
        reverse=True,
    )
    largest_gap, lower_end, upper_start = gaps[0]
    split = (lower_end + upper_start) // 2

    conn.execute(f"""
        CREATE TEMP TABLE labeled AS
        SELECT *, CASE WHEN final_update_id < {split} THEN 'low_id' ELSE 'high_id' END AS family
        FROM messages
    """)

    family_stats = records(conn, """
        SELECT family,
               count(*) AS messages,
               sum(rows_per_message) AS rows,
               min(final_update_id) AS min_update_id,
               max(final_update_id) AS max_update_id,
               min(last_update_id) AS min_cross_seq,
               max(last_update_id) AS max_cross_seq,
               min(received_time) AS first_received_ns,
               max(received_time) AS last_received_ns,
               1000000000.0 * (count(*) - 1) / nullif(max(received_time) - min(received_time), 0) AS mean_messages_per_second,
               avg(rows_per_message) AS mean_rows,
               quantile_cont(rows_per_message, 0.50) AS p50_rows,
               quantile_cont(rows_per_message, 0.90) AS p90_rows,
               quantile_cont(rows_per_message, 0.99) AS p99_rows,
               max(rows_per_message) AS max_rows,
               max(bid_rows) AS max_bid_rows,
               max(ask_rows) AS max_ask_rows,
               count(*) FILTER (WHERE bid_rows = 50 OR ask_rows = 50) AS messages_hitting_50_side,
               count(*) FILTER (WHERE bid_rows = 200 OR ask_rows = 200) AS messages_hitting_200_side,
               count(*) FILTER (WHERE bid_rows = 1000 OR ask_rows = 1000) AS messages_hitting_1000_side,
               count(*) FILTER (WHERE rows_per_message > 100) AS messages_gt_100_rows,
               count(*) FILTER (WHERE rows_per_message > 400) AS messages_gt_400_rows,
               count(*) FILTER (WHERE rows_per_message > 1000) AS messages_gt_1000_rows
        FROM labeled
        GROUP BY family
        ORDER BY family
    """)

    cadence = records(conn, cadence_sql("labeled", "family"))

    id_continuity = records(conn, """
        WITH ordered AS (
            SELECT family, received_time, final_update_id, last_update_id,
                   final_update_id - lag(final_update_id) OVER (PARTITION BY family ORDER BY received_time) AS update_id_step,
                   last_update_id - lag(last_update_id) OVER (PARTITION BY family ORDER BY received_time) AS cross_seq_step
            FROM labeled
        )
        SELECT family,
               count(update_id_step) AS transitions,
               avg(CASE WHEN update_id_step = 1 THEN 1.0 ELSE 0.0 END) AS fraction_update_id_plus_one,
               min(update_id_step) AS min_update_id_step,
               quantile_cont(update_id_step, 0.50) AS p50_update_id_step,
               quantile_cont(update_id_step, 0.99) AS p99_update_id_step,
               max(update_id_step) AS max_update_id_step,
               quantile_cont(cross_seq_step, 0.50) AS p50_cross_seq_step,
               quantile_cont(cross_seq_step, 0.90) AS p90_cross_seq_step,
               quantile_cont(cross_seq_step, 0.99) AS p99_cross_seq_step
        FROM ordered
        WHERE update_id_step IS NOT NULL
        GROUP BY family
        ORDER BY family
    """)

    cross_family_alignment = records(conn, """
        WITH low AS (
            SELECT last_update_id,
                   min(received_time) AS received_time,
                   min(event_time) AS event_time,
                   min(transaction_time) AS transaction_time,
                   min(final_update_id) AS update_id,
                   max(rows_per_message) AS rows_per_message
            FROM labeled WHERE family = 'low_id'
            GROUP BY last_update_id
        ), high AS (
            SELECT last_update_id,
                   min(received_time) AS received_time,
                   min(event_time) AS event_time,
                   min(transaction_time) AS transaction_time,
                   min(final_update_id) AS update_id,
                   max(rows_per_message) AS rows_per_message
            FROM labeled WHERE family = 'high_id'
            GROUP BY last_update_id
        ), paired AS (
            SELECT low.last_update_id,
                   abs(low.received_time - high.received_time) / 1000000.0 AS receive_difference_ms,
                   abs(low.event_time - high.event_time) AS event_difference_ms,
                   abs(low.transaction_time - high.transaction_time) AS transaction_difference_ms,
                   low.rows_per_message AS low_rows,
                   high.rows_per_message AS high_rows
            FROM low JOIN high USING (last_update_id)
        )
        SELECT count(*) AS shared_cross_sequences,
               quantile_cont(receive_difference_ms, 0.50) AS receive_difference_p50_ms,
               quantile_cont(receive_difference_ms, 0.90) AS receive_difference_p90_ms,
               quantile_cont(receive_difference_ms, 0.99) AS receive_difference_p99_ms,
               max(receive_difference_ms) AS receive_difference_max_ms,
               avg(CASE WHEN transaction_difference_ms = 0 THEN 1.0 ELSE 0.0 END) AS same_transaction_time_fraction,
               avg(CASE WHEN event_difference_ms <= 1 THEN 1.0 ELSE 0.0 END) AS event_time_within_1ms_fraction
        FROM paired
    """)[0]

    transport_lag = records(conn, """
        SELECT family,
               quantile_cont(received_time / 1000000.0 - event_time, 0.01) AS event_lag_p01_ms,
               quantile_cont(received_time / 1000000.0 - event_time, 0.50) AS event_lag_p50_ms,
               quantile_cont(received_time / 1000000.0 - event_time, 0.90) AS event_lag_p90_ms,
               quantile_cont(received_time / 1000000.0 - event_time, 0.99) AS event_lag_p99_ms,
               quantile_cont(received_time / 1000000.0 - transaction_time, 0.50) AS transaction_lag_p50_ms,
               quantile_cont(received_time / 1000000.0 - transaction_time, 0.99) AS transaction_lag_p99_ms
        FROM labeled
        GROUP BY family
        ORDER BY family
    """)

    largest_by_family = records(conn, """
        SELECT family, final_update_id, last_update_id, received_time, event_time, transaction_time,
               rows_per_message, bid_rows, ask_rows
        FROM labeled
        QUALIFY row_number() OVER (PARTITION BY family ORDER BY rows_per_message DESC) <= 20
        ORDER BY family, rows_per_message DESC
    """)

    first_messages = records(conn, """
        SELECT family, received_time, event_time, transaction_time,
               final_update_id, last_update_id, rows_per_message, bid_rows, ask_rows
        FROM labeled ORDER BY received_time LIMIT 60
    """)

    return {
        "distinct_update_ids": len(unique_ids),
        "largest_update_id_gaps": [
            {"gap": gap, "lower": lower, "upper": upper} for gap, lower, upper in gaps[:20]
        ],
        "chosen_split": split,
        "family_stats": family_stats,
        "family_cadence": cadence,
        "id_continuity": id_continuity,
        "cross_family_alignment": cross_family_alignment,
        "transport_lag": transport_lag,
        "largest_messages_by_family": largest_by_family,
        "first_messages": first_messages,
    }


def analyze_binance(conn: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    conn.execute("""
        CREATE TEMP TABLE messages AS
        SELECT received_time, event_time, transaction_time,
               first_update_id, final_update_id, prev_final_update_id,
               count(*) AS rows_per_message,
               count(*) FILTER (WHERE lower(CAST(side AS VARCHAR)) = 'bid') AS bid_rows,
               count(*) FILTER (WHERE lower(CAST(side AS VARCHAR)) = 'ask') AS ask_rows
        FROM p
        GROUP BY received_time, event_time, transaction_time,
                 first_update_id, final_update_id, prev_final_update_id
    """)

    continuity = records(conn, """
        WITH ordered AS (
            SELECT *,
                   lag(final_update_id) OVER (ORDER BY received_time) AS prior_final_update_id,
                   first_update_id - prev_final_update_id AS first_minus_prev,
                   final_update_id - first_update_id + 1 AS update_id_span
            FROM messages
        )
        SELECT count(*) AS messages,
               count(prior_final_update_id) AS comparable_transitions,
               avg(CASE WHEN prev_final_update_id = prior_final_update_id THEN 1.0 ELSE 0.0 END)
                   FILTER (WHERE prior_final_update_id IS NOT NULL) AS pu_matches_prior_u_fraction,
               count(*) FILTER (WHERE prior_final_update_id IS NOT NULL AND prev_final_update_id != prior_final_update_id) AS pu_mismatches,
               quantile_cont(first_minus_prev, 0.50) AS p50_first_minus_prev,
               quantile_cont(update_id_span, 0.50) AS p50_update_id_span,
               quantile_cont(update_id_span, 0.90) AS p90_update_id_span,
               quantile_cont(update_id_span, 0.99) AS p99_update_id_span,
               max(update_id_span) AS max_update_id_span,
               corr(rows_per_message, update_id_span) AS rows_span_correlation
        FROM ordered
    """)[0]

    cadence = records(conn, cadence_sql("messages"))[0]
    transport_lag = records(conn, """
        SELECT quantile_cont(received_time / 1000000.0 - event_time, 0.01) AS event_lag_p01_ms,
               quantile_cont(received_time / 1000000.0 - event_time, 0.50) AS event_lag_p50_ms,
               quantile_cont(received_time / 1000000.0 - event_time, 0.90) AS event_lag_p90_ms,
               quantile_cont(received_time / 1000000.0 - event_time, 0.99) AS event_lag_p99_ms,
               quantile_cont(received_time / 1000000.0 - transaction_time, 0.50) AS transaction_lag_p50_ms,
               quantile_cont(received_time / 1000000.0 - transaction_time, 0.99) AS transaction_lag_p99_ms
        FROM messages
    """)[0]

    size_stats = records(conn, """
        SELECT count(*) AS messages,
               avg(rows_per_message) AS mean_rows,
               quantile_cont(rows_per_message, 0.50) AS p50_rows,
               quantile_cont(rows_per_message, 0.90) AS p90_rows,
               quantile_cont(rows_per_message, 0.99) AS p99_rows,
               max(rows_per_message) AS max_rows,
               max(bid_rows) AS max_bid_rows,
               max(ask_rows) AS max_ask_rows
        FROM messages
    """)[0]

    first_messages = records(conn, """
        SELECT received_time, event_time, transaction_time,
               first_update_id, final_update_id, prev_final_update_id,
               rows_per_message, bid_rows, ask_rows
        FROM messages ORDER BY received_time LIMIT 30
    """)

    return {
        "continuity": continuity,
        "cadence": cadence,
        "transport_lag": transport_lag,
        "message_size": size_stats,
        "first_messages": first_messages,
    }


def markdown(payload: dict[str, Any]) -> str:
    bybit = payload["bybit"]
    binance = payload["binance_futures"]
    lines = ["# CryptoHFTData deep probe", "", "## Bybit stream families", ""]
    for row in bybit["family_stats"]:
        cadence = next(item for item in bybit["family_cadence"] if item["family"] == row["family"])
        lines.extend([
            f"### {row['family']}",
            f"- Messages: `{row['messages']:,}`; mean rate: `{float(row['mean_messages_per_second']):.3f}/s`",
            f"- Update ID range: `{row['min_update_id']}`–`{row['max_update_id']}`",
            f"- Receive cadence P50/P90/P99: `{cadence['receive_p50_ms']}` / `{cadence['receive_p90_ms']}` / `{cadence['receive_p99_ms']}` ms",
            f"- Rows P50/P90/P99/max: `{row['p50_rows']}` / `{row['p90_rows']}` / `{row['p99_rows']}` / `{row['max_rows']}`",
            f"- Max bid/ask rows: `{row['max_bid_rows']}` / `{row['max_ask_rows']}`",
            f"- Messages hitting 50/200/1000 rows on one side: `{row['messages_hitting_50_side']}` / `{row['messages_hitting_200_side']}` / `{row['messages_hitting_1000_side']}`",
            "",
        ])
    alignment = bybit["cross_family_alignment"]
    lines.extend([
        "### Cross-family alignment",
        f"- Shared cross-sequence values: `{alignment['shared_cross_sequences']:,}`",
        f"- Receive-time separation P50/P90/P99: `{alignment['receive_difference_p50_ms']}` / `{alignment['receive_difference_p90_ms']}` / `{alignment['receive_difference_p99_ms']}` ms",
        f"- Same transaction timestamp fraction: `{alignment['same_transaction_time_fraction']}`",
        "",
        "## Binance futures diff-depth continuity",
        f"- Messages: `{binance['continuity']['messages']:,}`",
        f"- `pu == previous u` fraction: `{binance['continuity']['pu_matches_prior_u_fraction']}`",
        f"- Mismatches: `{binance['continuity']['pu_mismatches']}`",
        f"- Receive cadence P50/P90/P99: `{binance['cadence']['receive_p50_ms']}` / `{binance['cadence']['receive_p90_ms']}` / `{binance['cadence']['receive_p99_ms']}` ms",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bybit_source = next(args.raw_dir.glob("bybit_*.parquet.zst"))
    binance_source = next(args.raw_dir.glob("binance_futures_*.parquet.zst"))

    with tempfile.TemporaryDirectory(prefix="chd-deep-") as tmp:
        tmp_path = Path(tmp)
        bybit_parquet = tmp_path / "bybit.parquet"
        binance_parquet = tmp_path / "binance.parquet"
        decompress(bybit_source, bybit_parquet)
        decompress(binance_source, binance_parquet)

        bybit_conn = make_connection(bybit_parquet)
        binance_conn = make_connection(binance_parquet)
        payload = {
            "bybit": analyze_bybit(bybit_conn),
            "binance_futures": analyze_binance(binance_conn),
        }
        bybit_conn.close()
        binance_conn.close()

    (args.out_dir / "deep_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=safe) + "\n",
        encoding="utf-8",
    )
    text = markdown(payload) + "\n"
    (args.out_dir / "deep_summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
