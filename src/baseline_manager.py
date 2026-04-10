"""
Machine learning feature scaling and baseline fallback logic.
"""
import pandas as pd
from src.api_extractors import _classify_pitch_family
from src.features import add_contextual_features

def apply_baseline_to_df(df: pd.DataFrame, baseline: dict, is_train: bool = False) -> pd.DataFrame:
    """Applies baseline tendency dictionaries to a DataFrame securely, unifying offline and online features."""
    df = df.copy()
    
    # Global
    if baseline.get('global'):
        df_global = pd.DataFrame.from_dict(baseline['global'], orient='index')
        df_global.index.name = 'pitcher_id'
        df = df.merge(df_global, on='pitcher_id', how='left')
    
    # Count
    if baseline.get('count'):
        df_count = pd.DataFrame.from_dict(baseline['count'], orient='index')
        if not df_count.empty:
            df_count.index.names = ['pitcher_id', 'balls', 'strikes']
            df = df.merge(df_count, on=['pitcher_id', 'balls', 'strikes'], how='left')
    
    # Batter Count
    if baseline.get('batter_count'):
        df_bcount = pd.DataFrame.from_dict(baseline['batter_count'], orient='index')
        if not df_bcount.empty:
            df_bcount.index.names = ['batter_id', 'balls', 'strikes']
            df = df.merge(df_bcount, on=['batter_id', 'balls', 'strikes'], how='left')
    
    # League count
    if baseline.get('league_count'):
        df_lcount = pd.DataFrame.from_dict(baseline['league_count'], orient='index')
        if not df_lcount.empty:
            df_lcount.index.names = ['balls', 'strikes']
            df = df.merge(df_lcount, on=['balls', 'strikes'], how='left')
    
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

    # Final fallback for any remaining NaNs (e.g. if global is somehow missing)
    tend_cols = [c for c in df.columns if c.startswith('tendency_') and c.endswith('_pct')]
    df[tend_cols] = df[tend_cols].fillna(0.3333)
    
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
                        df[pct_col] = new_pct.fillna(0.3333)
    
    # Add platoon advantage
    if 'pitcher_hand' in df.columns and 'batter_side' in df.columns:
        df['is_platoon_advantage'] = ((df['pitcher_hand'] == df['batter_side']) & (df['batter_side'] != 'S')).astype(int)
    
    return df


