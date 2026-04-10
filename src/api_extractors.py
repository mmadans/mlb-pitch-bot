"""
Feature engineering from raw MLB API game JSON.
"""
from src.constants import FASTBALL_CODES, BREAKING_CODES, OFFSPEED_CODES

def _classify_pitch_family(code: str | None) -> str:
    """Map pitch type code to family: Fastball, Breaking, Offspeed, or Other."""
    if not isinstance(code, str) or not code:
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


def extract_pitches_with_context(play_data: dict, game_date: str | None = None) -> list[dict]:
    """
    Extract pitch-level rows with score at time of pitch and previous pitch in AB.
    play_data: raw game JSON from MLB API.
    game_date: optional date string (YYYY-MM-DD). If not provided, tries to extract from play_data.
    """
    all_plays = play_data.get("liveData", {}).get("plays", {}).get("allPlays", [])
    pitches = []
    home_score, away_score = 0, 0
    pitcher_pitch_counts = {}
    matchup_tracker = {}

    for play in all_plays:
        if "playEvents" not in play:
            result = play.get("result", {})
            home_score = result.get("homeScore", home_score)
            away_score = result.get("awayScore", away_score)
            continue

        matchup = play.get("matchup", {})
        batter_info = matchup.get("batter") or {}
        batter_name = batter_info.get("fullName", "")
        batter_id = batter_info.get("id")
        
        pitcher_info = matchup.get("pitcher") or {}
        pitcher_name = pitcher_info.get("fullName", "")
        pitcher_id = pitcher_info.get("id")

        about = play.get("about", {})
        inning = about.get("inning") or 0
        half = about.get("halfInning", "")

        at_bat_index = about.get("atBatIndex")
        pitch_events = [e for e in play["playEvents"] if e.get("isPitch")]
        prev_pitch_type_in_ab = None
        prev_pitch_call_in_ab = None
        prev_pX_in_ab = 0.0
        prev_pZ_in_ab = 0.0
        
        if pitcher_id and batter_id:
            matchup_tracker[(pitcher_id, batter_id)] = matchup_tracker.get((pitcher_id, batter_id), 0) + 1
        times_faced = matchup_tracker.get((pitcher_id, batter_id), 1)
        
        # Streak tracking for the current at-bat
        breaking_streak = 0
        fastball_streak = 0
        offspeed_streak = 0

        if not game_date:
            game_date = play_data.get("gameData", {}).get("datetime", {}).get("officialDate", "")
        venue_id = play_data.get("gameData", {}).get("venue", {}).get("id")
        for event in pitch_events:
            details = event.get("details", {})
            type_info = details.get("type", {})
            code = type_info.get("code")
            pitch_family = _classify_pitch_family(code)
            
            pitch_data = event.get("pitchData", {})
            count = event.get("count", {})
            
            if pitcher_id:
                pitcher_pitch_counts[pitcher_id] = pitcher_pitch_counts.get(pitcher_id, 0) + 1
            pitch_count = pitcher_pitch_counts.get(pitcher_id, 1)

            pitch = {
                "at_bat_index": at_bat_index,
                "pitch_index": event.get("index"),
                "game_date": game_date,
                "batter": batter_name,
                "batter_id": batter_id,
                "pitcher": pitcher_name,
                "pitcher_id": pitcher_id,
                "pitch_type": code,
                "pitch_family": pitch_family,
                "pitcher_hand": matchup.get("pitchHand", {}).get("code"),
                "batter_side": matchup.get("batSide", {}).get("code"),
                "men_on_base": matchup.get("splits", {}).get("menOnBase"),
                "call": details.get("description"),
                "velocity": pitch_data.get("startSpeed"),
                "spin_rate": (pitch_data.get("breaks") or {}).get("spinRate"),
                "zone": pitch_data.get("zone"),
                "pX": (pitch_data.get("coordinates") or {}).get("pX"),
                "pZ": (pitch_data.get("coordinates") or {}).get("pZ"),
                "pitch_type_desc": type_info.get("description"),
                "inning": inning,
                "half_inning": half,
                "balls": count.get("balls"),
                "strikes": count.get("strikes"),
                "outs": count.get("outs"),
                "score_home": home_score,
                "score_away": away_score,
                "park_id": venue_id,
                "game_pk": play_data.get("gamePk"),
                "game_type": play_data.get("gameData", {}).get("game", {}).get("type"),
                "prev_pitch_type_in_ab": prev_pitch_type_in_ab,
                "prev_pitch_call": prev_pitch_call_in_ab,
                "prev_pX": prev_pX_in_ab,
                "prev_pZ": prev_pZ_in_ab,
                "pitch_count_in_game": pitch_count,
                "times_faced_today": times_faced,
                "breaking_streak": breaking_streak,
                "fastball_streak": fastball_streak,
                "offspeed_streak": offspeed_streak,
            }
            pitches.append(pitch)
            
            # Update streaks for the NEXT pitch
            if pitch_family == "Breaking":
                breaking_streak += 1
                fastball_streak = 0
                offspeed_streak = 0
            elif pitch_family == "Fastball":
                fastball_streak += 1
                breaking_streak = 0
                offspeed_streak = 0
            elif pitch_family == "Offspeed":
                offspeed_streak += 1
                breaking_streak = 0
                fastball_streak = 0
            else:
                # 'Other' resest all streaks
                breaking_streak = 0
                fastball_streak = 0
                offspeed_streak = 0

            prev_pitch_type_in_ab = (code or "UN").upper() if code else "UN"
            prev_pitch_call_in_ab = details.get("description")
            try:
                prev_pX_in_ab = float((pitch_data.get("coordinates") or {}).get("pX") or 0.0)
                prev_pZ_in_ab = float((pitch_data.get("coordinates") or {}).get("pZ") or 0.0)
            except (ValueError, TypeError):
                prev_pX_in_ab = 0.0
                prev_pZ_in_ab = 0.0

        result = play.get("result", {})
        home_score = result.get("homeScore", home_score)
        away_score = result.get("awayScore", away_score)

    return pitches


# All (balls, strikes) count combinations for one-hot encoding
