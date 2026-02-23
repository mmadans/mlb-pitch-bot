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
from src.database import save_pitches_to_db


def get_date_range(days: int) -> tuple[str, str]:
    """Return (start_date, end_date) for the past N days in YYYY-MM-DD."""
    end = datetime.now()
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _extract_all_pitches_from_games(games: list[dict]) -> pd.DataFrame:
    """Extract all pitch events from a list of games into a single DataFrame."""
    all_dfs = []
    for i, game in enumerate(games):
        game_id = game.get("game_id")
        summary = game.get("summary", f"game {game_id}")
        print(f"    [{i+1}/{len(games)}] Extracting pitches from {summary}")
        try:
            play_data = statsapi.get("game", {"gamePk": game_id})
            pitch_rows = extract_pitches_with_context(play_data)
            if pitch_rows:
                df = pd.DataFrame(pitch_rows)
                df["game_pk"] = game_id
                all_dfs.append(df)
        except Exception as e:
            print(f"      Skip: {e}")

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrate the addition of all features to the raw pitch data.
    """
    print("    Adding situational features...")
    df = add_pitcher_count_tendencies(df)
    df = add_global_pitcher_tendencies(df)
    df = add_contextual_features(df)
    return df


def build_dataset(
    start_date: str,
    end_date: str,
    to_db: bool = True,
) -> int:
    """
    Fetch all final games in the date range, build pitch features,
    and save to SQLite DB.
    Processes day-by-day to be memory efficient.
    """
    current_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    total_pitches = 0

    while current_dt <= end_dt:
        d_str = current_dt.strftime("%Y-%m-%d")
        print(f"Processing games for {d_str}...")
        
        games = statsapi.schedule(date=d_str)
        final_games = [g for g in games if g.get("status") == "Final"]
        
        if final_games:
            day_df = _extract_all_pitches_from_games(final_games)
            if not day_df.empty:
                print(f"    Collected {len(day_df)} pitches.")
                if to_db:
                    save_pitches_to_db(day_df)
                
                total_pitches += len(day_df)
        
        current_dt += timedelta(days=1)

    if total_pitches == 0:
        print("No pitch data collected.")
        return 0

    print(f"Finished! Total pitches collected: {total_pitches}")
    return total_pitches


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build pitch-feature dataset from MLB API.")
    p.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    p.add_argument("--days", type=int, default=None, help="Number of recent days to scrape (if --start not set)")
    args = p.parse_args()
    
    start, end = args.start, args.end
    if not start:
        days = args.days or 7
        start, end = get_date_range(days)
    
    build_dataset(start, end, to_db=True)
