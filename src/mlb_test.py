import statsapi
import pandas as pd

def get_game_pbp(game_pk):
    """
    Fetches play-by-play data for a specific game.
    """
    # 487637 is Game 7 of the 2016 World Series
    game_data = statsapi.get('game', {'gamePk': game_pk})
    all_plays = game_data['liveData']['plays']['allPlays']
    
    pitch_data = []
    
    for play in all_plays:
        # We only care about plays with pitch data
        if 'playEvents' not in play:
            continue
            
        batter = play['matchup']['batter']['fullName']
        pitcher = play['matchup']['pitcher']['fullName']
        
        for event in play['playEvents']:
            if event['isPitch']:
                # Extract relevant features for your future ML model
                pitch = {
                    'batter': batter,
                    'pitcher': pitcher,
                    'pitch_type': event['details'].get('type', {}).get('code'),
                    'call': event['details'].get('description'),
                    'velocity': event['pitchData'].get('startSpeed'),
                    'spin_rate': event['pitchData'].get('breaks', {}).get('spinRate'),
                    'inning': play['about']['inning'],
                    'balls': event['count']['balls'],
                    'strikes': event['count']['strikes'],
                    'outs': event['count']['outs']
                }
                pitch_data.append(pitch)
                
    return pd.DataFrame(pitch_data)

if __name__ == "__main__":
    print("Fetching data with uv...")
    df = get_game_pbp(487637)
    print(f"Successfully fetched {len(df)} pitches.")
    print(df.head())