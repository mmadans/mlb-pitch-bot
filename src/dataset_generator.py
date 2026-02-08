"""
Pull pitch data for all games from the last week and build a feature dataset.
Run with: uv run python -m src.dataset_generator
  or:     uv run python src/dataset_generator.py  (from project root)
"""
import sys
from pathlib import Path

# Ensure project root is on path when running as script
_root = Path(__file__).resolve().parent.parent
if _root not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(_root))

from datetime import datetime, timedelta

import statsapi

from src.features import build_pitch_features


def get_last_week_dates() -> tuple[str, str]:
    """Return (start_date, end_date) for the past 7 days in YYYY-MM-DD."""
    end = datetime.now()
    start = end - timedelta(days=7)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def build_last_week_dataset(
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: str = "data",
) -> str:
    """
    Fetch all final games in the date range, build pitch features per game,
    concatenate into one DataFrame, and save to CSV.

    Returns the path to the saved CSV.
    """
    if start_date is None or end_date is None:
        start_date, end_date = get_last_week_dates()

    print(f"Fetching schedule from {start_date} to {end_date}...")
    games = statsapi.schedule(start_date=start_date, end_date=end_date)
    final_games = [g for g in games if g.get("status") == "Final"]

    if not final_games:
        print("No final games in range.")
        return ""

    all_dfs = []
    for i, game in enumerate(final_games):
        game_id = game.get("game_id") or game.get("id")
        summary = game.get("summary", f"game {game_id}")
        print(f"  [{i+1}/{len(final_games)}] {summary}")
        try:
            play_data = statsapi.get("game", {"gamePk": game_id})
            df = build_pitch_features(play_data)
            if not df.empty:
                df["game_pk"] = game_id
                all_dfs.append(df)
        except Exception as e:
            print(f"    Skip: {e}")

    if not all_dfs:
        print("No pitch data collected.")
        return ""

    import os
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    combined = pd.concat(all_dfs, ignore_index=True)
    path = os.path.join(output_dir, f"pitch_features_{start_date}_to_{end_date}.csv")
    combined.to_csv(path, index=False)
    print(f"Saved {len(combined)} pitches to {path}")
    return path


if __name__ == "__main__":
    build_last_week_dataset()
