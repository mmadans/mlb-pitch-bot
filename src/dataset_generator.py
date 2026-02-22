"""
Pull pitch data for all games from the last week and build a feature dataset.

Run with uv (from project root):
  uv run python -m src.dataset_generator
"""

from datetime import datetime, timedelta
import os

import pandas as pd
import statsapi

from src.features import (
    add_contextual_features,
    add_global_pitcher_tendencies,
    add_pitcher_count_tendencies,
    extract_pitches_with_context,
)


def get_last_week_dates() -> tuple[str, str]:
    """Return (start_date, end_date) for the past 7 days in YYYY-MM-DD."""
    end = datetime.now()
    start = end - timedelta(days=7)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _extract_all_pitches_from_games(games: list[dict]) -> pd.DataFrame:
    """Extract all pitch events from a list of games into a single DataFrame."""
    all_dfs = []
    for i, game in enumerate(games):
        game_id = game.get("game_id")
        summary = game.get("summary", f"game {game_id}")
        print(f"  [{i+1}/{len(games)}] Extracting pitches from {summary}")
        try:
            play_data = statsapi.get("game", {"gamePk": game_id})
            pitch_rows = extract_pitches_with_context(play_data)
            if pitch_rows:
                df = pd.DataFrame(pitch_rows)
                df["game_pk"] = game_id
                all_dfs.append(df)
        except Exception as e:
            print(f"    Skip: {e}")

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate the addition of all features to the raw pitch data.
    """
    print("Adding global pitcher tendencies...")
    df = add_global_pitcher_tendencies(df)
    
    print("Adding situation-specific pitcher tendencies (by count)...")
    df = add_pitcher_count_tendencies(df)

    print("Adding contextual features (count, leverage, etc.)...")
    df = add_contextual_features(df)
    return df


def build_last_week_dataset(
    start_date: str | None = None,
    end_date: str | None = None,
    output_dir: str = "data",
) -> str:
    """
    Fetch all final games in the date range, build pitch features,
    and save to CSV.
    """
    if start_date is None or end_date is None:
        start_date, end_date = get_last_week_dates()

    print(f"Fetching schedule from {start_date} to {end_date}...")
    games = statsapi.schedule(start_date=start_date, end_date=end_date)
    final_games = [g for g in games if g.get("status") == "Final"]

    if not final_games:
        print("No final games in range.")
        return ""

    # 1. Extract all pitches from all games into a single DataFrame
    all_pitches_df = _extract_all_pitches_from_games(final_games)

    if all_pitches_df.empty:
        print("No pitch data collected.")
        return ""

    # 2. Add features
    features_df = add_features(all_pitches_df)

    # 3. Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"pitch_features_{start_date}_to_{end_date}.csv")
    features_df.to_csv(path, index=False)
    print(f"Saved {len(features_df)} pitches to {path}")
    return path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build pitch-feature dataset from MLB API.")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD (default: 7 days ago)")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    p.add_argument("--out", type=str, default="data", help="Output directory (default: data)")
    args = p.parse_args()
    build_last_week_dataset(
        start_date=args.start,
        end_date=args.end,
        output_dir=args.out,
    )
