import sqlite3
import pandas as pd
import json
import os
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data/pitches.db"
OUTPUT_DIR = ROOT / "dashboard/public/data"

def export_dashboard_data():
    """Extracts interesting stats from sqlite and saves as JSON for the React app."""
    print("Exporting data for Web Dashboard...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH}")
        return
        
    conn = sqlite3.connect(DB_PATH)
    
    # Load recent pitches
    df = pd.read_sql("SELECT * FROM pitches", conn)
    conn.close()
    
    if df.empty:
        print("Database is empty.")
        return
        
    print(f"Loaded {len(df)} total pitches.")
    
    # 1. Summary Stats
    total_pitches = len(df)
    total_games = df['game_pk'].nunique() if 'game_pk' in df.columns else 0
    k_count = len(df[df['events'] == 'strikeout']) if 'events' in df.columns else 0
    
    # 2. Pitch Group Distribution
    # Group similar pitches to match our classifier families
    fastballs = {"FF", "FT", "FC", "SI", "FS", "SF", "FA", "ST"}
    breaking = {"SL", "CU", "KC", "SV", "CS", "CB", "GY"}
    offspeed = {"CH", "SC", "FO", "KN", "EP"}
    
    def classify_pitch(code):
        if not code or pd.isna(code): return "Unknown"
        c = str(code).upper()
        if c in fastballs: return "Fastball"
        if c in breaking: return "Breaking"
        if c in offspeed: return "Offspeed"
        return "Other"
        
    df['pitch_family'] = df['pitch_type'].apply(classify_pitch)
    family_counts = df[df['pitch_family'] != "Unknown"]['pitch_family'].value_counts().to_dict()
    
    # 3. Top Pitchers by Usage
    top_pitchers = df['pitcher'].value_counts().head(10).to_dict()
    
    # 4. Generate some "Mock" Surprisal data for the UI since we don't save the inference scores to the DB yet
    # We will grab 10 random strikeouts and assign them a random surprisal score to demonstrate the UI
    strikeouts = df[df['events'] == 'strikeout'].copy() if 'events' in df.columns else pd.DataFrame()
    surprises = []
    
    if not strikeouts.empty:
        sample_size = min(15, len(strikeouts))
        sample_ks = strikeouts.sample(sample_size, random_state=42)
        
        for idx, row in sample_ks.iterrows():
            import random
            surprisal = round(random.uniform(2.5, 6.0), 2) # High surprisal range
            prob = max(0.01, round(2**(-surprisal), 3)) 
            
            surprises.append({
                "id": str(idx),
                "pitcher": row.get('pitcher', 'Unknown'),
                "batter_id": row.get('batter_id', 'Unknown'),
                "pitch_type": row.get('pitch_type', 'Unknown'),
                "pitch_family": row.get('pitch_family', 'Unknown'),
                "speed": round(row.get('release_speed', 0), 1),
                "count": f"{row.get('balls', 0)}-{row.get('strikes', 0)}",
                "inning": f"{row.get('half_inning', '')} {row.get('inning', '')}",
                "prob": prob,
                "surprisal": surprisal,
                "outcome": "Strikeout"
            })
            
    # Sort surprises by highest surprisal
    surprises = sorted(surprises, key=lambda x: x['surprisal'], reverse=True)
    
    # Compile the final data object
    dashboard_data = {
        "last_updated": pd.Timestamp.now().isoformat(),
        "summary": {
            "total_pitches": total_pitches,
            "total_games": total_games,
            "total_strikeouts": k_count
        },
        "pitch_distribution": family_counts,
        "top_pitchers": [{"name": k, "count": v} for k, v in top_pitchers.items()],
        "top_surprises": surprises
    }
    
    # Save to the React public folder so it can be fetched client-side
    out_file = OUTPUT_DIR / "dashboard_data.json"
    with open(out_file, "w") as f:
        json.dump(dashboard_data, f, indent=2)
        
    print(f"Dashboard data exported successfully to {out_file}")

if __name__ == "__main__":
    export_dashboard_data()
