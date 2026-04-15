"""
Machine learning feature scaling and baseline fallback logic.
"""
import pandas as pd
from src.api_extractors import _classify_pitch_family
from src.features import add_contextual_features

def apply_baseline_to_df(df: pd.DataFrame, baseline: dict, is_train: bool = False) -> pd.DataFrame:
    """Applies baseline tendency dictionaries to a DataFrame securely, unifying offline and online features."""
    df = df.copy()

    # Use empirical priors computed from training data; fall back to approximate MLB averages
    league_priors = baseline.get('league_priors', {
        'Fastball': 0.55, 'Breaking': 0.25, 'Offspeed': 0.20
    })
    
    # Merge each tendency dict: (baseline_key, index_column_names)
    _tendency_merges = [
        ('global',       ['pitcher_id']),
        ('count',        ['pitcher_id', 'balls', 'strikes']),
        ('batter_count', ['batter_id',  'balls', 'strikes']),
        ('league_count', ['balls', 'strikes']),
    ]
    for key, join_cols in _tendency_merges:
        if not baseline.get(key):
            continue
        tendency_df = pd.DataFrame.from_dict(baseline[key], orient='index')
        if tendency_df.empty:
            continue
        tendency_df.index.names = join_cols
        df = df.merge(tendency_df, on=join_cols, how='left')
    
    # Out pitch
    if baseline.get('out_pitch'):
        out_pitch_s = pd.Series(baseline['out_pitch'], name='primary_out_pitch')
        df = df.merge(out_pitch_s, left_on='pitcher_id', right_index=True, how='left')
    
    if "primary_out_pitch" not in df.columns:
        df["primary_out_pitch"] = "Fastball"
    df["primary_out_pitch"] = df["primary_out_pitch"].fillna("Fastball")
    
    # Add contextual features to ensure pitch_family and other base dependencies exist
    df = add_contextual_features(df)
    if 'pitch_family' not in df.columns and 'pitch_type' in df.columns:
        df['pitch_family'] = df['pitch_type'].apply(_classify_pitch_family)

    # For missing tendencies, fallback intelligently rather than defaulting to 0.3333
    if not is_train:
        for fam in ["Fastball", "Breaking", "Offspeed", "Other"]:
            # Fallback hierarchy: Pitcher's Count -> Pitcher's Global -> 0.3333
            if f"tendency_count_{fam}_pct" in df.columns and f"tendency_global_{fam}_pct" in df.columns:
                df[f"tendency_count_{fam}_pct"] = df[f"tendency_count_{fam}_pct"].fillna(df[f"tendency_global_{fam}_pct"])
                
            # Fallback hierarchy: Batter's Count -> League Average Count -> 0.3333
            if f"tendency_batter_count_{fam}_pct" in df.columns and f"tendency_league_count_{fam}_pct" in df.columns:
                df[f"tendency_batter_count_{fam}_pct"] = df[f"tendency_batter_count_{fam}_pct"].fillna(df[f"tendency_league_count_{fam}_pct"])

    # Final fallback for any remaining NaNs — use empirical league priors per family
    for fam, prior in league_priors.items():
        for col in df.columns:
            if col.startswith('tendency_') and col.endswith('_pct') and f'_{fam}_pct' in col:
                df[col] = df[col].fillna(prior)
    # Any remaining tendency columns (e.g. Other family) fall back to uniform
    remaining_tend_cols = [c for c in df.columns if c.startswith('tendency_') and c.endswith('_pct') and df[c].isna().any()]
    df[remaining_tend_cols] = df[remaining_tend_cols].fillna(1/3)
    
    if is_train and "pitch_family" in df.columns:
        # Prevent Target Leakage using Leave-One-Out (LOO) calculation for train data
        for prefix in ["tendency_global", "tendency_count", "tendency_batter_count", "tendency_league_count"]:
            total_col = f"{prefix}_total_pitches"
            if total_col in df.columns:
                for fam in ["Fastball", "Breaking", "Offspeed"]:
                    pct_col = f"{prefix}_{fam}_pct"
                    if pct_col in df.columns:
                        # Inverse calculation for Laplace smoothing:
                        # pct = (actual_count + 1) / (total + num_classes)
                        num_classes = 3
                        # We use rounding to combat floating point precision issues
                        actual_count = (df[pct_col] * (df[total_col] + num_classes) - 1.0).round()
                        
                        # Subtract 1 if this row's family matches the column's family
                        is_match = (df["pitch_family"] == fam).astype(int)
                        new_count = actual_count - is_match
                        new_total = df[total_col] - 1
                        
                        # Clip to prevent negative counts in edge cases
                        new_count = new_count.clip(lower=0)
                        new_total = new_total.clip(lower=0)
                        
                        # Re-apply Laplace Smoothing
                        new_pct = (new_count + 1.0) / (new_total + num_classes)
                        df[pct_col] = new_pct.fillna(league_priors.get(fam, 1/3))
    
    # Delta tendency features: how much this pitcher deviates from league average at this count.
    # This gives XGBoost a pre-computed "pitcher quirk" signal rather than making it discover
    # the difference between two separate tendency columns on its own.
    for fam in ["Fastball", "Breaking", "Offspeed"]:
        count_col = f"tendency_count_{fam}_pct"
        league_col = f"tendency_league_count_{fam}_pct"
        global_col = f"tendency_global_{fam}_pct"
        if count_col in df.columns and league_col in df.columns:
            df[f"delta_count_{fam}"] = df[count_col] - df[league_col]
        if global_col in df.columns and league_col in df.columns:
            df[f"delta_global_{fam}"] = df[global_col] - df[league_col]

    # Add platoon advantage (switch hitters always bat from the advantaged side)
    if 'pitcher_hand' in df.columns and 'batter_side' in df.columns:
        same_hand = (df['pitcher_hand'] == df['batter_side']) & (df['batter_side'] != 'S')
        switch_hitter = df['batter_side'] == 'S'
        df['is_platoon_advantage'] = (same_hand | switch_hitter).astype(int)
    
    return df


