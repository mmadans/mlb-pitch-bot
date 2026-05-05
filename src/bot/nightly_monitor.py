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

    for col in [
        "prob_fastball",
        "prob_breaking",
        "prob_offspeed",
        "surprisal",
        "pitcher_sample_n",
        "count_sample_n",
    ]:
        if col in lp.columns:
            lp[col] = pd.to_numeric(lp[col], errors="coerce")

    lp["date"] = lp["timestamp"].dt.date.astype(str)
    lp["probs_valid"] = (
        lp[["prob_fastball", "prob_breaking", "prob_offspeed"]].notna().all(axis=1)
    )
    lp["surprisal_finite"] = lp["surprisal"].notna() & np.isfinite(lp["surprisal"])

    prob_cols = lp[["prob_fastball", "prob_breaking", "prob_offspeed"]]
    family_map = {
        "prob_fastball": "Fastball",
        "prob_breaking": "Breaking",
        "prob_offspeed": "Offspeed",
    }
    lp["predicted_family"] = np.where(
        lp["probs_valid"],
        prob_cols.fillna(0).idxmax(axis=1).map(family_map),
        pd.NA,
    )
    lp["correct"] = (lp["predicted_family"] == lp["actual_pitch_family"]).astype(
        "Int64"
    )
    return lp


FAMILIES = ["Fastball", "Breaking", "Offspeed"]
PROB_COLS = ["prob_fastball", "prob_breaking", "prob_offspeed"]
N_CAL_BINS = 5  # keep bins wide enough to have data on a single game day


def compute_error_breakdown(valid_day: pd.DataFrame) -> pd.DataFrame:
    """
    For each actual pitch family, compute what fraction the model predicted
    as each family. Rows = actual, columns = predicted_as_*.
    Perfect model = 100% on the diagonal.
    """
    rows = []
    for actual in FAMILIES:
        actual_rows = valid_day[valid_day["actual_pitch_family"] == actual]
        total = len(actual_rows)
        row = {"actual_family": actual, "n": total}
        for pred in FAMILIES:
            frac = (
                (actual_rows["predicted_family"] == pred).sum() / total
                if total
                else float("nan")
            )
            row[f"predicted_as_{pred.lower()}_pct"] = round(frac * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_calibration(valid_day: pd.DataFrame) -> pd.DataFrame:
    """
    Reliability diagram data: for each pitch family, bin the predicted
    probability into N_CAL_BINS buckets and compute actual frequency per bucket.
    Perfect calibration = predicted_prob ≈ actual_freq (points on the diagonal).
    """
    bins = np.linspace(0, 1, N_CAL_BINS + 1)
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    rows = []
    for fam, col in zip(FAMILIES, PROB_COLS):
        y_true = (valid_day["actual_pitch_family"] == fam).astype(float)
        y_pred = valid_day[col]
        bin_indices = np.digitize(y_pred, bins, right=True).clip(1, N_CAL_BINS) - 1
        for i, mid in enumerate(bin_midpoints):
            mask = bin_indices == i
            count = mask.sum()
            rows.append(
                {
                    "family": fam,
                    "bin_midpoint": round(float(mid), 2),
                    "mean_predicted": round(float(y_pred[mask].mean()), 4)
                    if count
                    else float("nan"),
                    "actual_freq": round(float(y_true[mask].mean()), 4)
                    if count
                    else float("nan"),
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def compute_surprisal_histogram(valid_day: pd.DataFrame):
    """wandb.Histogram of surprisal values for the day."""
    finite = valid_day[np.isfinite(valid_day["surprisal"].fillna(float("nan")))]
    if len(finite) == 0:
        return None
    return wandb.Histogram(finite["surprisal"].tolist())


def compute_pitcher_errors(valid_day: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher prediction accuracy, sorted by error rate descending."""
    if "pitcher_name" not in valid_day.columns or valid_day.empty:
        return pd.DataFrame()
    grp = (
        valid_day.groupby("pitcher_name")
        .agg(
            predictions=("correct", "count"),
            correct=("correct", "sum"),
        )
        .reset_index()
    )
    grp["errors"] = grp["predictions"] - grp["correct"]
    grp["error_rate"] = (grp["errors"] / grp["predictions"]).round(3)
    return grp.sort_values("error_rate", ascending=False).reset_index(drop=True)


def compute_count_accuracy(valid_day: pd.DataFrame) -> pd.DataFrame:
    """
    Accuracy broken down by count situation.
    Requires balls/strikes columns; returns empty DataFrame if not present.
    """
    if "balls" not in valid_day.columns or "strikes" not in valid_day.columns:
        return pd.DataFrame()
    df = valid_day.dropna(subset=["balls", "strikes"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["count_str"] = (
        df["balls"].astype(int).astype(str)
        + "-"
        + df["strikes"].astype(int).astype(str)
    )
    rows = []
    # All individual counts
    for count, grp in df.groupby("count_str"):
        rows.append(
            {
                "situation": count,
                "n": len(grp),
                "accuracy": round(grp["correct"].mean(), 3),
            }
        )
    # Aggregate situations
    situations = {
        "2-strike": df["strikes"].astype(int) == 2,
        "3-ball": df["balls"].astype(int) == 3,
        "hitter_ahead": df["balls"].astype(int) > df["strikes"].astype(int),
        "pitcher_ahead": df["strikes"].astype(int) > df["balls"].astype(int),
        "even": df["balls"].astype(int) == df["strikes"].astype(int),
        "first_pitch": (df["balls"].astype(int) == 0)
        & (df["strikes"].astype(int) == 0),
    }
    for label, mask in situations.items():
        grp = df[mask]
        if len(grp):
            rows.append(
                {
                    "situation": label,
                    "n": len(grp),
                    "accuracy": round(grp["correct"].mean(), 3),
                }
            )
    return pd.DataFrame(rows).sort_values("situation").reset_index(drop=True)


def compute_sample_size_dist(valid_day: pd.DataFrame) -> pd.DataFrame:
    """
    Distribution of pitcher_sample_n values in buckets.
    Shows how often the model is operating on sparse vs rich pitcher history.
    """
    if "pitcher_sample_n" not in valid_day.columns or valid_day.empty:
        return pd.DataFrame()
    buckets = [75, 100, 150, 200, 300, float("inf")]
    labels = ["75-100", "100-150", "150-200", "200-300", "300+"]
    rows = []
    s = valid_day["pitcher_sample_n"].dropna()
    for i, label in enumerate(labels):
        lo, hi = buckets[i], buckets[i + 1]
        count = ((s >= lo) & (s < hi)).sum()
        rows.append({"sample_bucket": label, "count": int(count)})
    return pd.DataFrame(rows)


def compute_metrics(
    lp: pd.DataFrame, log_date: str
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day = lp[lp["date"] == log_date].copy()
    valid_day = day[day["probs_valid"]]
    finite_day = valid_day[valid_day["surprisal_finite"]]

    # Brier score: sum of per-class MSE (standard multi-class Brier)
    brier = float("nan")
    if len(valid_day):
        brier = sum(
            (
                (
                    valid_day[col]
                    - (valid_day["actual_pitch_family"] == fam).astype(float)
                )
                ** 2
            ).mean()
            for fam, col in zip(FAMILIES, PROB_COLS)
        )

    metrics = {
        "total_predictions": len(day),
        "finite_surprisal_rate": day["surprisal_finite"].mean()
        if len(day)
        else float("nan"),
        "overall_accuracy": valid_day["correct"].mean()
        if len(valid_day)
        else float("nan"),
        "brier_score": brier,
        "tweet_rate": (finite_day["surprisal"] > SURPRISAL_THRESHOLD).mean()
        if len(finite_day)
        else float("nan"),
        "mean_surprisal": finite_day["surprisal"].mean()
        if len(finite_day)
        else float("nan"),
    }

    for fam, col in zip(FAMILIES, PROB_COLS):
        cls = fam.lower()
        actual_rows = valid_day[valid_day["actual_pitch_family"] == fam]
        actual_freq = (
            len(actual_rows) / len(valid_day) if len(valid_day) else float("nan")
        )
        mean_pred = valid_day[col].mean() if len(valid_day) else float("nan")
        metrics[f"recall_{cls}"] = (
            actual_rows["correct"].mean() if len(actual_rows) else float("nan")
        )
        metrics[f"mean_pred_prob_{cls}"] = mean_pred
        metrics[f"calibration_gap_{cls}"] = mean_pred - actual_freq

    error_df = compute_error_breakdown(valid_day) if len(valid_day) else pd.DataFrame()
    cal_df = compute_calibration(valid_day) if len(valid_day) else pd.DataFrame()
    pitcher_df = compute_pitcher_errors(valid_day) if len(valid_day) else pd.DataFrame()
    sample_df = (
        compute_sample_size_dist(valid_day) if len(valid_day) else pd.DataFrame()
    )
    count_acc_df = (
        compute_count_accuracy(valid_day) if len(valid_day) else pd.DataFrame()
    )
    surprisal_hist = compute_surprisal_histogram(valid_day) if len(valid_day) else None

    return (
        metrics,
        day,
        error_df,
        cal_df,
        pitcher_df,
        sample_df,
        count_acc_df,
        surprisal_hist,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date",
        default=None,
        help="Date to log (YYYY-MM-DD). Defaults to most recent.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute metrics but do not push to W&B."
    )
    args = parser.parse_args()

    lp = load_predictions(DATABASE_PATH)
    if lp.empty:
        print("No predictions found in database.")
        sys.exit(0)

    log_date = args.date or lp["date"].max()
    run_name = f"nightly-{log_date}"
    dry_run = args.dry_run

    print(f"Logging date  : {log_date}")
    print(f"W&B project   : {os.getenv('WANDB_PROJECT')}")
    print(f"Dry run       : {dry_run}")

    (
        metrics,
        day,
        error_df,
        cal_df,
        pitcher_df,
        sample_df,
        count_acc_df,
        surprisal_hist,
    ) = compute_metrics(lp, log_date)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:<35} {v:.4f}" if isinstance(v, float) else f"  {k:<35} {v}")

    if not error_df.empty:
        print("\nError breakdown (% predicted as each family):")
        print(error_df.to_string(index=False))

    if not cal_df.empty:
        print("\nCalibration (predicted prob vs actual frequency):")
        print(cal_df.to_string(index=False))

    if not pitcher_df.empty:
        print("\nPer-pitcher error rate (top 10):")
        print(pitcher_df.head(10).to_string(index=False))

    if not sample_df.empty:
        print("\nPitcher sample size distribution:")
        print(sample_df.to_string(index=False))

    if not count_acc_df.empty:
        print("\nAccuracy by count situation:")
        print(count_acc_df.to_string(index=False))

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

        table_cols = [
            "timestamp",
            "pitcher_id",
            "pitcher_name",
            "actual_pitch_family",
            "predicted_family",
            "prob_fastball",
            "prob_breaking",
            "prob_offspeed",
            "surprisal",
            "correct",
            "balls",
            "strikes",
            "pitcher_sample_n",
            "count_sample_n",
        ]
        available = [c for c in table_cols if c in day.columns]
        valid_day = day[day["probs_valid"] & day["predicted_family"].notna()]
        fam_idx = {f: i for i, f in enumerate(FAMILIES)}
        confusion_matrix = wandb.plot.confusion_matrix(
            y_true=[fam_idx.get(f, 0) for f in valid_day["actual_pitch_family"]],
            preds=[fam_idx.get(f, 0) for f in valid_day["predicted_family"]],
            class_names=FAMILIES,
        )
        to_log = {
            "predictions": wandb.Table(dataframe=day[available].reset_index(drop=True)),
            "error_breakdown": wandb.Table(dataframe=error_df),
            "calibration": wandb.Table(dataframe=cal_df),
            "confusion_matrix": confusion_matrix,
        }
        if not pitcher_df.empty:
            to_log["pitcher_errors"] = wandb.Table(dataframe=pitcher_df)
        if not sample_df.empty:
            to_log["sample_size_distribution"] = wandb.Table(dataframe=sample_df)
        if not count_acc_df.empty:
            to_log["count_accuracy"] = wandb.Table(dataframe=count_acc_df)
        if surprisal_hist is not None:
            to_log["surprisal_distribution"] = surprisal_hist
        run.log(to_log)
        run.finish()
        print(f"\nLogged to W&B: {run_name}")
    else:
        print("\nDry run — nothing pushed to W&B.")


if __name__ == "__main__":
    main()
