PITCH_ABBR = {
    "Four-Seam Fastball": "FF",
    "4-Seam": "FF",
    "Sinker": "SI",
    "Cutter": "FC",
    "Slider": "SL",
    "Sweeper": "ST",
    "Curveball": "CU",
    "Knuckle Curve": "KC",
    "Changeup": "CH",
    "Splitter": "FS",
    "Slurve": "SV",
    "Knuckleball": "KN",
    "Forkball": "FO",
    "Eephus": "EP",
    "Screwball": "SC",
    "Other": "OT",
    "Unknown": "UN"
}

def get_pitch_abbr(full_name: str) -> str:
    """Returns a short abbreviation for a pitch type."""
    if not full_name:
        return "UN"
    return PITCH_ABBR.get(full_name, full_name[:3].upper())
