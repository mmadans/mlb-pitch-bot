"""
Batter tendency processing: Fetching season stats from MLB API and calculating 
advanced metrics (Whiff, Chase) from historically collected pitch data.
"""
import pandas as pd
import statsapi
from src.data.api_extractors import _classify_pitch_family

_SWING_CALLS = {"swinging strike", "foul", "foul tip", "in play", "swinging strike (blocked)"}
_WHIFF_CALLS = {"swinging strike", "swinging strike (blocked)"}

def fetch_batter_season_stats(batter_id: int, season: int = 2024) -> dict:
    """
    Fetches standard season stats for a batter via MLB Stats API.
    Returns a dict with OBP, K%, BB%.
    """
    try:
        stats_data = statsapi.player_stat_data(batter_id, group="hitting", type="season", season=season)
        if not stats_data or not stats_data.get("stats"):
            return {"obp": 0.0, "k_pct": 0.0, "bb_pct": 0.0}
        
        s = stats_data["stats"][0]["stats"]
        pa = s.get("plateAppearances", 0)
        if pa == 0:
            return {"obp": 0.0, "k_pct": 0.0, "bb_pct": 0.0}
        
        return {
            "obp": float(s.get("obp", 0.0)),
            "k_pct": s.get("strikeOuts", 0) / pa,
            "bb_pct": s.get("baseOnBalls", 0) / pa
        }
    except Exception:
        return {"obp": 0.0, "k_pct": 0.0, "bb_pct": 0.0}

def calculate_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates Whiff Rate and Chase Rate from pitch-level data.
    - Whiff Rate: Swinging Strikes / Swings
    - Chase Rate: Swings on pitches outside the zone / Total pitches outside the zone
    
    'zone' 1-9 is inside, 11-14 is outer.
    'call' description based.
    """
    if df.empty:
        return pd.DataFrame()

    call_lower = df["call"].fillna("").str.lower()
    df["is_swing"] = call_lower.apply(lambda x: any(s in x for s in _SWING_CALLS))
    df["is_whiff"] = call_lower.apply(lambda x: any(w in x for w in _WHIFF_CALLS))
    
    # Chase Rate: zone > 9 is outside (11, 12, 13, 14, or null)
    # Note: StatsAPI zones 1-9 are strike zone. 11-14 are chase zones.
    df["is_outside_zone"] = df["zone"].apply(lambda z: z > 9 if pd.notnull(z) else True)
    df["is_chase"] = (df["is_swing"] & df["is_outside_zone"])
    
    # Aggregates
    batter_stats = df.groupby("batter_id").agg(
        total_swings=("is_swing", "sum"),
        total_whiffs=("is_whiff", "sum"),
        total_outside=("is_outside_zone", "sum"),
        total_chases=("is_chase", "sum")
    ).reset_index()
    
    batter_stats["whiff_rate"] = (batter_stats["total_whiffs"] / batter_stats["total_swings"]).fillna(0)
    batter_stats["chase_rate"] = (batter_stats["total_chases"] / batter_stats["total_outside"]).fillna(0)
    
    return batter_stats[["batter_id", "whiff_rate", "chase_rate"]]

def calculate_whiff_by_pitch_family(df: pd.DataFrame) -> pd.DataFrame:
    """
    Whiff rate for each batter broken down by pitch family.
    """
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    df["pitch_family"] = df["pitch_type"].apply(_classify_pitch_family)
    
    call_lower = df["call"].fillna("").str.lower()
    df["is_swing"] = call_lower.apply(lambda x: any(s in x for s in _SWING_CALLS))
    df["is_whiff"] = call_lower.apply(lambda x: any(w in x for w in _WHIFF_CALLS))
    
    group = df.groupby(["batter_id", "pitch_family"]).agg(
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum")
    ).reset_index()
    
    group["whiff_rate"] = (group["whiffs"] / group["swings"]).fillna(0)
    
    # Pivot to columns
    pivot = group.pivot(index="batter_id", columns="pitch_family", values="whiff_rate").reset_index()
    pivot.columns = [f"whiff_rate_{col}" if col != "batter_id" else col for col in pivot.columns]
    
    return pivot.fillna(0)

def get_batter_features(df: pd.DataFrame, use_api: bool = True) -> pd.DataFrame:
    """
    Orchestrates building a feature set for hitters.
    Combines API season stats and DB-calculated metrics.
    """
    # 1. DB Metrics (Whiff, Chase)
    advanced = calculate_advanced_metrics(df)
    whiff_fam = calculate_whiff_by_pitch_family(df)
    
    batter_features = advanced.merge(whiff_fam, on="batter_id", how="outer")
    
    if use_api:
        # 2. API Metrics (OBP, K%, BB%)
        # For performance, only fetch unique batters
        unique_batters = df[["batter_id", "batter"]].drop_duplicates()
        api_stats = []
        for _, row in unique_batters.iterrows():
            if pd.isnull(row["batter_id"]): continue
            print(f"      Fetching API stats for {row['batter']}...")
            stats = fetch_batter_season_stats(int(row["batter_id"]))
            stats["batter_id"] = int(row["batter_id"])
            api_stats.append(stats)
        
        if api_stats:
            api_df = pd.DataFrame(api_stats)
            batter_features = batter_features.merge(api_df, on="batter_id", how="outer")
            
    return batter_features.fillna(0)
