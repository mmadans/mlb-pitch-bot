"""
Nightly monitoring script — logs daily prediction health to W&B.

Run after the last game of the day:
    uv run python -m src.nightly_monitor
    uv run python -m src.nightly_monitor --date 2026-04-14   # specific date
    uv run python -m src.nightly_monitor --dry-run           # preview only
"""
import os
import sys
import argparse
import sqlite3
import numpy as np
import pandas as pd
import wandb
from pathlib import Path
from dotenv import load_dotenv

from src.constants import DATABASE_PATH, SURPRISAL_THRESHOLD

load_dotenv()


def load_predictions(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        lp = pd.read_sql_query(
            """
            SELECT lp.*, p.pitcher AS pitcher_name
            FROM live_predictions lp
            LEFT JOIN (SELECT DISTINCT pitcher_id, pitcher FROM pitches) p
                   ON lp.pitcher_id = p.pitcher_id
            ORDER BY lp.timestamp
            """,
            conn,
            parse_dates=["timestamp"],
        )

    for col in ["prob_fastball", "prob_breaking", "prob_offspeed", "surprisal",
                "pitcher_sample_n", "count_sample_n"]:
        if col in lp.columns:
            lp[col] = pd.to_numeric(lp[col], errors="coerce")

    lp["date"] = lp["timestamp"].dt.date.astype(str)
    lp["probs_valid"] = lp[["prob_fastball", "prob_breaking", "prob_offspeed"]].notna().all(axis=1)
    lp["surprisal_finite"] = lp["surprisal"].notna() & np.isfinite(lp["surprisal"])

    prob_cols = lp[["prob_fastball", "prob_breaking", "prob_offspeed"]]
    family_map = {"prob_fastball": "Fastball", "prob_breaking": "Breaking", "prob_offspeed": "Offspeed"}
    lp["predicted_family"] = np.where(
        lp["probs_valid"],
        prob_cols.fillna(0).idxmax(axis=1).map(family_map),
        pd.NA,
    )
    lp["correct"] = (lp["predicted_family"] == lp["actual_pitch_family"]).astype("Int64")
    return lp


def compute_metrics(lp: pd.DataFrame, log_date: str) -> dict:
    day = lp[lp["date"] == log_date].copy()
    valid_day = day[day["probs_valid"]]
    finite_day = valid_day[valid_day["surprisal_finite"]]

    metrics = {
        "total_predictions":    len(day),
        "valid_prob_rate":       day["probs_valid"].mean(),
        "finite_surprisal_rate": day["surprisal_finite"].mean() if len(day) else float("nan"),
        "overall_accuracy":      valid_day["correct"].mean() if len(valid_day) else float("nan"),
        "tweet_rate":            (finite_day["surprisal"] > SURPRISAL_THRESHOLD).mean() if len(finite_day) else float("nan"),
        "mean_surprisal":        finite_day["surprisal"].mean() if len(finite_day) else float("nan"),
    }

    for fam, prob_col in [("Fastball", "prob_fastball"), ("Breaking", "prob_breaking"), ("Offspeed", "prob_offspeed")]:
        fam_rows = valid_day[valid_day["actual_pitch_family"] == fam]
        actual_freq = (valid_day["actual_pitch_family"] == fam).mean() if len(valid_day) else float("nan")
        mean_pred = valid_day[prob_col].mean() if len(valid_day) else float("nan")
        metrics[f"accuracy_{fam.lower()}"]        = fam_rows["correct"].mean() if len(fam_rows) else float("nan")
        metrics[f"mean_pred_prob_{fam.lower()}"]  = mean_pred
        metrics[f"calibration_gap_{fam.lower()}"] = mean_pred - actual_freq

    return metrics, day


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",    default=None, help="Date to log (YYYY-MM-DD). Defaults to most recent.")
    parser.add_argument("--dry-run", action="store_true", help="Compute metrics but do not push to W&B.")
    args = parser.parse_args()

    lp = load_predictions(DATABASE_PATH)
    if lp.empty:
        print("No predictions found in database.")
        sys.exit(0)

    log_date = args.date or lp["date"].max()
    run_name = f"nightly-{log_date}"
    dry_run  = args.dry_run

    print(f"Logging date  : {log_date}")
    print(f"W&B project   : {os.getenv('WANDB_PROJECT')}")
    print(f"Dry run       : {dry_run}")

    metrics, day = compute_metrics(lp, log_date)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:<35} {v:.4f}" if isinstance(v, float) else f"  {k:<35} {v}")

    if len(day) == 0:
        print(f"\nNo predictions found for {log_date}. Nothing to log.")
        sys.exit(0)

    if not dry_run:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "mlb-pitch-bot"),
            name=run_name,
            job_type="nightly-monitoring",
            resume="allow",
        )
        run.log(metrics)

        table_cols = ["timestamp", "pitcher_id", "pitcher_name", "actual_pitch_family",
                      "predicted_family", "prob_fastball", "prob_breaking", "prob_offspeed",
                      "surprisal", "correct", "pitcher_sample_n", "count_sample_n"]
        available = [c for c in table_cols if c in day.columns]
        run.log({"predictions": wandb.Table(dataframe=day[available].reset_index(drop=True))})
        run.finish()
        print(f"\nLogged to W&B: {run_name}")
    else:
        print("\nDry run — nothing pushed to W&B.")


if __name__ == "__main__":
    main()
