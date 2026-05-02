import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from src.bot.utils import get_pitch_abbr
from src.bot.signal_labels import SIGNAL_LABELS
from src.constants import PITCH_COLORS
from src.data.api_extractors import _classify_pitch_family

_PITCH_SHORT_NAME = {
    "Four-Seam Fastball": "4-Seam",
    "Two-Seam Fastball": "2-Seam",
    "Sinker": "Sinker",
    "Cutter": "Cutter",
    "Slider": "Slider",
    "Sweeper": "Sweeper",
    "Curveball": "Curve",
    "Knuckle Curve": "Kn. Curve",
    "Changeup": "Changeup",
    "Splitter": "Splitter",
    "Forkball": "Fork",
    "Knuckleball": "Knuckle",
    "Eephus": "Eephus",
}

_CALL_ABBR = {
    "Ball": "Ball", "Called Strike": "Called K", "Swinging Strike": "Swing K",
    "Foul": "Foul", "Foul Tip": "Foul Tip",
    "In play, out(s)": "In Play", "In play, run(s)": "In Play", "In play, no out": "In Play",
    "Hit By Pitch": "HBP",
}


def set_dark_theme(ax):
    ax.set_facecolor('#111111')
    ax.tick_params(colors='#AAAAAA')
    for spine in ax.spines.values():
        spine.set_color('#333333')
    ax.xaxis.label.set_color('#AAAAAA')
    ax.yaxis.label.set_color('#AAAAAA')
    ax.title.set_color('#FFFFFF')


def _draw_header(ax, pitch_data):
    ax.set_facecolor('#111111')
    ax.axis('off')

    p_hand = pitch_data.get("pitcher_hand", "")
    pitcher = f"{pitch_data.get('pitcher', 'Pitcher')}" + (f" ({p_hand})" if p_hand else "")
    b_side = pitch_data.get("batter_side", "")
    batter = f"{pitch_data.get('batter', 'Batter')}" + (f" ({b_side})" if b_side else "")

    inning = pitch_data.get("inning", 1)
    half = pitch_data.get("half_inning", "Top").title()
    balls, strikes, outs = pitch_data.get("balls", 0), pitch_data.get("strikes", 0), pitch_data.get("outs", 0)
    a_team, h_team = pitch_data.get("away_team", "AWY"), pitch_data.get("home_team", "HOM")
    score_str = f"{a_team} {pitch_data.get('score_away', 0)}, {h_team} {pitch_data.get('score_home', 0)}"
    runners_str = f"Bases: {pitch_data.get('men_on_base', 'Empty').replace('_', ' ')}"

    ax.text(0.02, 0.7, pitcher.upper(), color='#FFFFFF', fontsize=22, fontweight='bold')
    ax.text(0.02, 0.3, f"vs {batter.upper()}", color='#FFFFFF', fontsize=18, fontweight='bold', style='italic')
    ax.text(0.98, 0.8, f"{a_team} @ {h_team}  |  {score_str}", color='#FFFFFF', fontsize=18, fontweight='bold', ha='right')
    ax.text(0.98, 0.45, f"{half.capitalize()} {inning}  |  {outs} OUTS", color='#0074D9', fontsize=14, fontweight='bold', ha='right')
    ax.text(0.98, 0.15, f"{balls}-{strikes} COUNT  |  {runners_str.upper()}", color='#FFFFFF', fontsize=12, fontweight='bold', ha='right')


def _draw_actual_header(ax, pitch_data):
    ax.set_facecolor('#111111')
    ax.axis('off')
    actual_fam = pitch_data.get("pitch_family") or _classify_pitch_family(pitch_data.get("pitch_type", "UN"))
    desc = pitch_data.get("pitch_type_desc", actual_fam)
    velo = pitch_data.get("velocity")
    spin = pitch_data.get("spin_rate")
    pitch_color = PITCH_COLORS.get(actual_fam, '#FFFFFF')

    ax.text(0.5, 0.92, "ACTUAL", color='#888888', fontsize=12, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    ax.text(0.5, 0.60, actual_fam.upper(), color=pitch_color, fontsize=24, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    stats_parts = ([desc] if desc else []) + ([f"{velo} MPH"] if velo else []) + ([f"{spin} RPM"] if spin else [])
    if stats_parts:
        ax.text(0.5, 0.15, "  |  ".join(stats_parts), color='#AAAAAA', fontsize=12,
                ha='center', va='top', transform=ax.transAxes)


def _draw_predicted_header(ax, pitch_data, probs):
    ax.set_facecolor('#111111')
    ax.axis('off')
    predicted_fam = max(probs, key=probs.get)
    predicted_prob = probs[predicted_fam]
    pred_color = PITCH_COLORS.get(predicted_fam, '#FFFFFF')

    ax.text(0.5, 0.92, "PREDICTED", color='#888888', fontsize=12, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    ax.text(0.5, 0.60, predicted_fam.upper(), color=pred_color, fontsize=24, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    ax.text(0.5, 0.22, f"{predicted_prob * 100:.0f}% CONFIDENCE", color='#AAAAAA', fontsize=12,
            fontweight='bold', ha='center', va='top', transform=ax.transAxes)


def _draw_zone_only(ax, pitch_data, sequence):
    set_dark_theme(ax)
    actual_fam = pitch_data.get("pitch_family") or _classify_pitch_family(pitch_data.get("pitch_type", "UN"))
    pitch_color = PITCH_COLORS.get(actual_fam, '#FFFFFF')

    sz_width = 17 / 12
    zone = patches.Rectangle((-sz_width / 2, 1.5), sz_width, 2.0,
                              linewidth=2, edgecolor='#555555', facecolor='none', linestyle='dashed')
    ax.add_patch(zone)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(0.3, 4.2)
    ax.set_aspect('equal')
    ax.axis('off')

    for i, p in enumerate(sequence):
        if isinstance(p, dict) and p.get("pX") is not None and p.get("pZ") is not None:
            color = PITCH_COLORS.get(p.get("pitch_family"), "#888888")
            ax.scatter(p["pX"], p["pZ"], s=200, color=color, edgecolor='none', alpha=0.25, zorder=3)
            ax.text(p["pX"], p["pZ"], str(i + 1), color='white', ha='center', va='center',
                    fontsize=8, fontweight='bold', alpha=0.4, zorder=4)

    pX, pZ = pitch_data.get("pX"), pitch_data.get("pZ")
    if pX is not None and pZ is not None:
        ax.scatter(pX, pZ, s=500, color=pitch_color, edgecolor='white', linewidth=2, zorder=5)
        ax.text(pX, pZ, str(len(sequence) + 1), color='white', ha='center', va='center',
                fontsize=12, fontweight='bold', zorder=6)


def _build_signals(pitch_data, predicted_fam, balls, strikes):
    def _frac(val):
        if val is None:
            return None
        v = float(val)
        return v if v <= 1.0 else v / 100.0

    def _delta_str(delta):
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta * 100:.0f}%"

    candidates = []  # (score, label, impact_str | None)

    L = SIGNAL_LABELS

    streak = int(pitch_data.get(f"{predicted_fam.lower()}_streak", 0) or 0)
    if streak >= 3:
        candidates.append((10, L["streak_long"].format(n=streak, family=predicted_fam.lower()), None))
    elif streak == 2:
        candidates.append((6, L["streak_short"].format(family=predicted_fam.lower()), None))

    if strikes == 2:
        candidates.append((9, L["count_two_strike"], None))
    elif balls == 3:
        candidates.append((8, L["count_three_ball"], None))
    elif balls == 0 and strikes == 0:
        candidates.append((4, L["count_first_pitch"], None))
    elif balls > strikes:
        candidates.append((3, L["count_hitters"].format(balls=balls, strikes=strikes), None))

    primary = str(pitch_data.get("primary_out_pitch", "") or "")
    if primary == predicted_fam:
        candidates.append((7, L["primary_out_pitch"], None))

    gt = _frac(pitch_data.get(f"tendency_global_{predicted_fam}_pct"))
    ct = _frac(pitch_data.get(f"tendency_count_{predicted_fam}_pct"))
    bt = _frac(pitch_data.get(f"tendency_batter_count_{predicted_fam}_pct"))

    if ct is not None and gt is not None and ct >= gt + 0.08:
        label = L["count_tendency_elevated"].format(pct=f"{ct*100:.0f}", balls=balls, strikes=strikes)
        candidates.append((8, label, _delta_str(ct - gt)))
    elif gt is not None:
        label = L["count_tendency_baseline"].format(family=predicted_fam, pct=f"{gt*100:.0f}")
        candidates.append((4, label, None))

    if bt is not None and gt is not None and abs(bt - gt) >= 0.05:
        candidates.append((5, L["batter_tendency"].format(pct=f"{bt*100:.0f}", family=predicted_fam), _delta_str(bt - gt)))
    elif bt is not None:
        candidates.append((3, L["batter_tendency"].format(pct=f"{bt*100:.0f}", family=predicted_fam), None))

    if pitch_data.get("is_platoon_advantage") == 1:
        candidates.append((2, L["platoon_advantage"], None))

    candidates.sort(key=lambda x: -x[0])
    return [(label, impact) for _, label, impact in candidates[:3]]


def _draw_signals_panel(ax, pitch_data, probs):
    ax.set_facecolor('#111111')
    ax.axis('off')
    predicted_fam = max(probs, key=probs.get)
    balls = pitch_data.get("balls", 0)
    strikes = pitch_data.get("strikes", 0)
    signals = _build_signals(pitch_data, predicted_fam, balls, strikes)

    ax.text(0.0, 0.97, "TOP PREDICTION REASONS", color='#FFFFFF', fontsize=14,
            fontweight='bold', ha='left', va='top', transform=ax.transAxes)

    pitch_color = PITCH_COLORS.get(predicted_fam, '#FFFFFF')
    y = 0.77
    for i, (label, impact) in enumerate(signals, 1):
        ax.text(0.0, y, f"{i}.  {label}", color='#DDDDDD', fontsize=13,
                ha='left', va='top', transform=ax.transAxes)
        if impact is not None:
            ax.text(1.0, y, f"{impact} {predicted_fam}", color=pitch_color, fontsize=13,
                    fontweight='bold', ha='right', va='top', transform=ax.transAxes)
        y -= 0.16


def _draw_sequence_strip(ax, sequence, pitch_data):
    ax.set_facecolor('#111111')
    ax.axis('off')
    ax.set_title("PITCH SEQUENCE", fontsize=14, color='#FFFFFF', fontweight='bold', pad=8, loc='left')

    full_seq = []
    for p in sequence:
        if isinstance(p, dict):
            full_seq.append({"name": p.get("pitch_type_desc", p.get("pitch_type_code", "?")),
                             "family": p.get("pitch_family", "Unknown"), "outcome": p.get("call", "")})
        else:
            full_seq.append({"name": str(p), "family": "Unknown", "outcome": ""})
    full_seq.append({
        "name": pitch_data.get("pitch_type_desc", pitch_data.get("pitch_type", "?")),
        "family": pitch_data.get("pitch_family") or _classify_pitch_family(pitch_data.get("pitch_type", "UN")),
        "outcome": pitch_data.get("call", ""),
    })

    visible = full_seq[-14:]
    nv = len(visible)
    offset = len(full_seq) - nv
    ml, mr = 0.05, 0.25
    xs = [ml + i / max(nv - 1, 1) * (1 - ml - mr) for i in range(nv)]

    for i, (x, p_info) in enumerate(zip(xs, visible)):
        seq_num = offset + i + 1
        color = PITCH_COLORS.get(p_info["family"], '#CCCCCC')
        is_current = (i == nv - 1)
        ax.scatter([x], [0.60], s=500 if is_current else 250,
                   color=color, alpha=1.0 if is_current else 0.5,
                   edgecolor='white' if is_current else 'none', linewidth=1.5,
                   transform=ax.transAxes, zorder=2, clip_on=True)
        ax.text(x, 0.60, str(seq_num), color='white', ha='center', va='center',
                fontsize=8 if is_current else 7, fontweight='bold',
                transform=ax.transAxes, zorder=3)
        short_name = _PITCH_SHORT_NAME.get(p_info["name"], p_info["name"])
        ax.text(x, 0.40, short_name, color=color,
                fontsize=11, ha='right', va='top', rotation=45,
                transform=ax.transAxes,
                alpha=1.0 if is_current else 0.85)


def _draw_prediction_bars(ax, probs, actual_fam):
    set_dark_theme(ax)
    ax.set_title("MODEL PREDICTION", fontsize=14, color='#FFFFFF', fontweight='bold', pad=8, loc='left')
    sorted_probs = sorted(probs.items(), key=lambda x: x[1])
    p_labels = [("★ " if f == actual_fam else "") + f.upper() for f, v in sorted_probs]
    p_values = [x[1] * 100 for x in sorted_probs]
    p_colors = [PITCH_COLORS.get(f, '#FFFFFF') for f, v in sorted_probs]
    bars = ax.barh(p_labels, p_values, color=p_colors, height=0.5, alpha=0.8)
    ax.set_xlim(0, 100)
    for spine in ['right', 'top', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False)
    for tick in ax.get_yticklabels():
        if "★" in tick.get_text():
            tick.set_fontweight('bold')
            tick.set_color('#FFFFFF')
    ax.tick_params(left=False, labelsize=9)
    for bar, val in zip(bars, p_values):
        ax.text(val + 3, bar.get_y() + bar.get_height() / 2, f"{val:.0f}%",
                va='center', color='white', fontsize=10, fontweight='bold')


def generate_pitch_infographic(pitch_data: dict, probs: dict, surprisal: float, sequence: list = None, output_path: str = "temp_plot.png") -> str:
    if sequence is None:
        sequence = []

    fig = plt.figure(figsize=(14, 9), facecolor='#111111')
    gs = GridSpec(4, 2,
                  height_ratios=[1.2, 1.3, 3.6, 1.5],
                  width_ratios=[1.2, 1],
                  hspace=0.08, wspace=0.10,
                  left=0.02, right=0.98, top=0.97, bottom=0.03)

    _draw_header(fig.add_subplot(gs[0, :]), pitch_data)
    _draw_actual_header(fig.add_subplot(gs[1, 0]), pitch_data)
    _draw_predicted_header(fig.add_subplot(gs[1, 1]), pitch_data, probs)
    _draw_zone_only(fig.add_subplot(gs[2, 0]), pitch_data, sequence)
    _draw_signals_panel(fig.add_subplot(gs[2, 1]), pitch_data, probs)
    _draw_sequence_strip(fig.add_subplot(gs[3, 0]), sequence, pitch_data)
    _draw_prediction_bars(fig.add_subplot(gs[3, 1]), probs, pitch_data.get("pitch_family", "Unknown"))

    fig.text(0.01, 0.01, "@PitchScript", color='#FFFFFF', alpha=0.2, fontsize=11, fontweight='bold')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#111111')
    plt.close()

    return output_path
