import statsapi
import pandas as pd
from datetime import datetime

def fetch_season_data(start_date='2025-05-01', end_date='2025-05-07'):
    """
    Fetches all pitches for a given date range. 
    Start small (one week) to test.
    """
    print(f"Fetching games from {start_date} to {end_date}...")
    games = statsapi.schedule(start_date=start_date, end_date=end_date)
    all_game_pbp = []

    for game in games:
        if game['status'] == 'Final':
            print(f"Processing: {game['summary']}")
            # Use our existing logic to get PBP
            df = get_game_pbp(game['game_id']) # You can import your function from mlb_test
            all_game_pbp.append(df)
            
    final_df = pd.concat(all_game_pbp, ignore_index=True)
    final_df.to_csv(f"data/training_data_{start_date}.csv", index=False)
    return final_df