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
#   primary_out_pitch       (none)
#   count_tendency_elevated {pct}, {balls}, {strikes}
#   count_tendency_baseline {family}, {pct}
#   batter_tendency         {pct}, {family}
#   platoon_advantage       (none)

SIGNAL_LABELS = {
    "streak_long":              "{n}-pitch {family} streak",
    "streak_short":             "2-pitch {family} streak",
    "count_two_strike":         "Two-strike count",
    "count_three_ball":         "3-ball count",
    "count_first_pitch":        "First pitch of at-bat",
    "count_hitters":            "Hitter's count ({balls}-{strikes})",
    "primary_out_pitch":        "His primary out pitch",
    "count_tendency_elevated":  "Count tendency: {pct}% at {balls}-{strikes}",
    "count_tendency_baseline":  "Throws {family} {pct}% overall",
    "batter_tendency":          "Batter sees {family} {pct}% of the time",
    "platoon_advantage":        "Platoon advantage",
}
