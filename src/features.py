"""
Feature engineering from raw MLB API game JSON.
Computes Pitcher's Tendency based on the last 5 pitches.
"""

from collections import Counter
from typing import Any


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
