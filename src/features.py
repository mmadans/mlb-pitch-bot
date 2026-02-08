"""
Feature engineering from raw MLB API game JSON.
Computes Pitcher's Tendency (last 5 pitches) and build_pitch_features
(count one-hot, is_leverage, previous pitch in AB).
"""

from collections import Counter

import pandas as pd


# Common pitch type groupings for tendency summary
FASTBALL_CODES = {"FF", "FT", "FC", "SI", "FS", "SF", "FA", "ST"}
BREAKING_CODES = {"SL", "CU", "KC", "SV", "CS", "CB", "GY"}
OFFSPEED_CODES = {"CH", "SC", "FO", "KN", "EP"}


def _classify_pitch_family(code: str | None) -> str:
    """Map pitch type code to family: Fastball, Breaking, Offspeed, or Other."""
    if not code:
        return "Other"
    code = code.upper()
    if code in FASTBALL_CODES:
        return "Fastball"
    if code in BREAKING_CODES:
        return "Breaking"
    if code in OFFSPEED_CODES:
        return "Offspeed"
    return "Other"


def _extract_pitches_in_order(game_json: dict) -> list[dict]:
    """
    Walk raw game JSON and yield pitch events in chronological order,
    each with play-level context (pitcher, batter, inning, count, etc.).
    """
    all_plays = game_json.get("liveData", {}).get("plays", {}).get("allPlays", [])
    pitches = []

    for play in all_plays:
        if "playEvents" not in play:
            continue
        matchup = play.get("matchup", {})
        batter = (matchup.get("batter") or {}).get("fullName", "")
        pitcher = (matchup.get("pitcher") or {}).get("fullName", "")
        about = play.get("about", {})
        inning = about.get("inning")
        half = about.get("halfInning", "")

        for event in play["playEvents"]:
            if not event.get("isPitch"):
                continue
            details = event.get("details", {})
            type_info = details.get("type", {})
            code = type_info.get("code")
            pitch_data = event.get("pitchData", {})
            count = event.get("count", {})

            pitch = {
                "batter": batter,
                "pitcher": pitcher,
                "pitch_type": code,
                "call": details.get("description"),
                "velocity": pitch_data.get("startSpeed"),
                "spin_rate": (pitch_data.get("breaks") or {}).get("spinRate"),
                "inning": inning,
                "half_inning": half,
                "balls": count.get("balls"),
                "strikes": count.get("strikes"),
                "outs": count.get("outs"),
            }
            pitches.append(pitch)

    return pitches


def _extract_pitches_with_context(play_data: dict) -> list[dict]:
    """
    Extract pitch-level rows with score at time of pitch and previous pitch in AB.
    play_data: raw game JSON from MLB API (e.g. statsapi.get('game', {'gamePk': pk})).
    """
    all_plays = play_data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    pitches = []
    home_score, away_score = 0, 0

    for play in all_plays:
        if "playEvents" not in play:
            result = play.get("result", {})
            home_score = result.get("homeScore", home_score)
            away_score = result.get("awayScore", away_score)
            continue

        matchup = play.get("matchup", {})
        batter = (matchup.get("batter") or {}).get("fullName", "")
        pitcher = (matchup.get("pitcher") or {}).get("fullName", "")
        about = play.get("about", {})
        inning = about.get("inning") or 0
        half = about.get("halfInning", "")

        pitch_events = [e for e in play["playEvents"] if e.get("isPitch")]
        prev_pitch_type_in_ab = None

        for event in pitch_events:
            details = event.get("details", {})
            type_info = details.get("type", {})
            code = type_info.get("code")
            pitch_data = event.get("pitchData", {})
            count = event.get("count", {})

            pitch = {
                "batter": batter,
                "pitcher": pitcher,
                "pitch_type": code,
                "call": details.get("description"),
                "velocity": pitch_data.get("startSpeed"),
                "spin_rate": (pitch_data.get("breaks") or {}).get("spinRate"),
                "inning": inning,
                "half_inning": half,
                "balls": count.get("balls"),
                "strikes": count.get("strikes"),
                "outs": count.get("outs"),
                "score_home": home_score,
                "score_away": away_score,
                "prev_pitch_type_in_ab": prev_pitch_type_in_ab,
            }
            pitches.append(pitch)
            prev_pitch_type_in_ab = (code or "UN").upper() if code else "UN"

        result = play.get("result", {})
        home_score = result.get("homeScore", home_score)
        away_score = result.get("awayScore", away_score)

    return pitches


# All (balls, strikes) count combinations for one-hot encoding
COUNT_LABELS = [
    "0-0", "0-1", "0-2", "1-0", "1-1", "1-2", "2-0", "2-1", "2-2",
    "3-0", "3-1", "3-2",
]


def build_pitch_features(play_data: dict) -> pd.DataFrame:
    """
    Build a pitch-level DataFrame from raw MLB API game JSON (play_data).

    Returns a DataFrame with:
    - Base pitch fields (batter, pitcher, pitch_type, velocity, inning, etc.)
    - One-hot encoding for current count (columns count_0_0, count_0_1, ... count_3_2)
    - is_leverage: 1 if inning > 7 and |home_score - away_score| <= 2, else 0
    - prev_pitch_type_in_ab: pitcher's previous pitch type in this at-bat (None on first pitch)
    """
    rows = _extract_pitches_with_context(play_data)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Count string and one-hot
    df["balls"] = df["balls"].fillna(0).astype(int).clip(0, 3)
    df["strikes"] = df["strikes"].fillna(0).astype(int).clip(0, 2)
    df["count_str"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)

    for label in COUNT_LABELS:
        df[f"count_{label.replace('-', '_')}"] = (df["count_str"] == label).astype(int)

    # Leverage: inning > 7 and score diff <= 2
    score_home = df["score_home"].fillna(0).astype(int)
    score_away = df["score_away"].fillna(0).astype(int)
    df["score_diff"] = (score_home - score_away).abs()
    inning = df["inning"].fillna(0).astype(int)
    df["is_leverage"] = ((inning > 7) & (df["score_diff"] <= 2)).astype(int)

    # Drop helper columns used only for feature construction
    df = df.drop(columns=["count_str", "score_diff"], errors="ignore")

    return df


def add_pitcher_tendency(game_json: dict) -> list[dict]:
    """
    Take raw game JSON from the MLB API and return a list of pitch dicts
    with Pitcher's Tendency features based on the last 5 pitches by that pitcher.

    Each pitch gets:
    - pitcher_tendency_primary: most common pitch type in last 5 (e.g. "FF", "SL").
    - pitcher_tendency_primary_pct: fraction of last 5 that were that type (0–1).
    - pitcher_tendency_fastball_pct: fraction of last 5 that were fastballs.
    - pitcher_tendency_breaking_pct: fraction of last 5 that were breaking.
    - pitcher_tendency_offspeed_pct: fraction of last 5 that were offspeed.
    - pitcher_tendency_pitches_used: number of pitches in the window (1–5).

    Pitches with fewer than 1 previous pitch from that pitcher have tendency
    fields set to None or 0.0 as appropriate.
    """
    pitches = _extract_pitches_in_order(game_json)
    # Per-pitcher rolling window of last 5 pitch type codes
    pitcher_history: dict[str, list[str]] = {}

    for p in pitches:
        pitcher = p["pitcher"]
        code = p.get("pitch_type") or "UN"
        code_upper = code.upper() if isinstance(code, str) else "UN"

        # Get last 5 for this pitcher (before this pitch)
        history = pitcher_history.get(pitcher, [])
        last_5 = history[-5:] if len(history) >= 5 else history

        # Compute tendency from last 5
        if not last_5:
            p["pitcher_tendency_primary"] = None
            p["pitcher_tendency_primary_pct"] = None
            p["pitcher_tendency_fastball_pct"] = None
            p["pitcher_tendency_breaking_pct"] = None
            p["pitcher_tendency_offspeed_pct"] = None
            p["pitcher_tendency_pitches_used"] = 0
        else:
            counts = Counter(last_5)
            primary, primary_count = counts.most_common(1)[0]
            n = len(last_5)
            p["pitcher_tendency_primary"] = primary
            p["pitcher_tendency_primary_pct"] = round(primary_count / n, 4)
            p["pitcher_tendency_fastball_pct"] = round(
                sum(1 for c in last_5 if _classify_pitch_family(c) == "Fastball") / n, 4
            )
            p["pitcher_tendency_breaking_pct"] = round(
                sum(1 for c in last_5 if _classify_pitch_family(c) == "Breaking") / n, 4
            )
            p["pitcher_tendency_offspeed_pct"] = round(
                sum(1 for c in last_5 if _classify_pitch_family(c) == "Offspeed") / n, 4
            )
            p["pitcher_tendency_pitches_used"] = n

        # Append this pitch to this pitcher's history (after computing tendency)
        if pitcher not in pitcher_history:
            pitcher_history[pitcher] = []
        pitcher_history[pitcher].append(code_upper)

    return pitches


if __name__ == "__main__":
    import statsapi
    import pandas as pd

    game_pk = 487637  # Game 7, 2016 World Series (same as mlb_test.py)
    print("Fetching game JSON...")
    game_data = statsapi.get("game", {"gamePk": game_pk})
    pitches_with_tendency = add_pitcher_tendency(game_data)
    df = pd.DataFrame(pitches_with_tendency)
    print(f"Loaded {len(df)} pitches with Pitcher's Tendency features.")
    tendency_cols = [
        "pitcher_tendency_primary",
        "pitcher_tendency_primary_pct",
        "pitcher_tendency_fastball_pct",
        "pitcher_tendency_breaking_pct",
        "pitcher_tendency_offspeed_pct",
        "pitcher_tendency_pitches_used",
    ]
    print(df[["pitcher", "pitch_type"] + tendency_cols].head(12).to_string())
