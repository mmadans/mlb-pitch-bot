# Display text templates for prediction reason signals.
# Edit these to change what appears in the "TOP PREDICTION REASONS" panel.
#
# Available placeholders per key:
#   streak_long             {n}, {family}
#   streak_short            {family}
#   count_two_strike        (none)
#   count_three_ball        (none)
#   count_first_pitch       (none)
#   count_hitters           {balls}, {strikes}
#   primary_out_pitch       {pitcher}
#   count_tendency_elevated {pct}, {balls}, {strikes}, {pitcher}
#   count_tendency_baseline {family}, {pct}, {pitcher}
#   batter_tendency         {pct}, {family}, {batter}
#   platoon_advantage       (none)

SIGNAL_LABELS = {
    "streak_long":              "{n}-pitch {family} streak",
    "streak_short":             "2-pitch {family} streak",
    "count_two_strike":         "Two-strike count",
    "count_three_ball":         "3-ball count",
    "count_first_pitch":        "First pitch of at-bat",
    "count_hitters":            "Hitter's count ({balls}-{strikes})",
    "primary_out_pitch":        "{pitcher}'s primary out pitch",
    "count_tendency_elevated":  "{pitcher} throws {family} {pct} at {balls}-{strikes}",
    "count_tendency_baseline":  "{pitcher} throws {family} {pct}% overall",
    "batter_tendency":          "{batter} sees {family} {pct}% of the time",
    "platoon_advantage":        "Platoon advantage",
}
