import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from src.bot.utils import get_pitch_abbr
from src.constants import PITCH_COLORS
from src.data.api_extractors import _classify_pitch_family

def set_dark_theme(ax):
    """Applies a premium dark theme to a matplotlib axis."""
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
    p_hand_str = f" ({p_hand})" if p_hand else ""
    pitcher = f"{pitch_data.get('pitcher', 'Pitcher')}{p_hand_str}"
    
    b_side = pitch_data.get("batter_side", "")
    b_side_str = f" ({b_side})" if b_side else ""
    batter = f"{pitch_data.get('batter', 'Batter')}{b_side_str}"
    
    inning = pitch_data.get("inning", 1)
    half = pitch_data.get("half_inning", "Top").title()
    balls = pitch_data.get("balls", 0)
    strikes = pitch_data.get("strikes", 0)
    outs = pitch_data.get("outs", 0)
    
    a_team = pitch_data.get("away_team", "AWY")
    h_team = pitch_data.get("home_team", "HOM")
    score_str = f"{a_team} {pitch_data.get('score_away', 0)}, {h_team} {pitch_data.get('score_home', 0)}"
    
    runners = pitch_data.get("men_on_base", "Empty")
    runners_str = f"Bases: {runners.replace('_', ' ')}"
    
    ax.text(0.02, 0.7, pitcher.upper(), color='#FFFFFF', fontsize=22, fontweight='bold')
    ax.text(0.02, 0.3, f"vs {batter.upper()}", color='#FFFFFF', fontsize=18, fontweight='bold', style='italic')
    
    ax.text(0.98, 0.8, f"{a_team} @ {h_team}  |  {score_str}", color='#FFFFFF', fontsize=18, fontweight='bold', ha='right')
    ax.text(0.98, 0.45, f"{half.capitalize()} {inning}  |  {outs} OUTS", color='#0074D9', fontsize=14, fontweight='bold', ha='right')
    ax.text(0.98, 0.15, f"{balls}-{strikes} COUNT  |  {runners_str.upper()}", color='#FFFFFF', fontsize=12, fontweight='bold', ha='right')

def _draw_sequence(ax, sequence, pitch_data):
    ax.set_facecolor('#111111')
    ax.axis('off')
    ax.set_title("SEQUENCE", fontsize=14, color='#FFFFFF', fontweight='bold', pad=10)
    
    if not sequence and not pitch_data:
        return
        
    full_seq = []
    for p in sequence:
        if isinstance(p, dict):
            full_seq.append({
                "name": p.get("pitch_type_desc", p.get("pitch_type_code", "Unknown")),
                "family": p.get("pitch_family", "Unknown"),
                "outcome": p.get("call", "")
            })
        else:
            full_seq.append({"name": str(p), "family": "Unknown", "outcome": ""})
            
    full_seq.append({
        "name": pitch_data.get("pitch_type_desc", pitch_data.get("pitch_type", "Unknown")),
        "family": pitch_data.get("pitch_family") or _classify_pitch_family(pitch_data.get("pitch_type", "UN")),
        "outcome": pitch_data.get("call", "")
    })
    
    spacing = 0.12
    start_y = 0.95
    
    for i, p_info in enumerate(full_seq):
        y = start_y - (i * spacing)
        if y < 0.05: break
        
        p_name = p_info["name"]
        p_fam = p_info["family"]
        p_outcome = p_info["outcome"]
        color = PITCH_COLORS.get(p_fam, '#CCCCCC')
        
        ax.scatter([0.15], [y], s=500, color=color, alpha=0.8, transform=ax.transAxes, zorder=2)
        ax.text(0.15, y, str(i+1), color='white', ha='center', va='center', fontsize=9, fontweight='bold', transform=ax.transAxes, zorder=3)
        ax.text(0.25, y, f"{get_pitch_abbr(p_name)}", color=color, fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.45, y, f"{p_outcome}", color='#FFFFFF', fontsize=11, style='italic', transform=ax.transAxes)

def _draw_strike_zone(ax, pitch_data, sequence, fig):
    set_dark_theme(ax)
    actual_fam = pitch_data.get("pitch_family") or _classify_pitch_family(pitch_data.get("pitch_type", "UN"))
    desc = pitch_data.get("pitch_type_desc", actual_fam)
    velo = pitch_data.get("velocity")
    spin = pitch_data.get("spin_rate")
    pitch_color = PITCH_COLORS.get(actual_fam, '#FFFFFF')
    
    ax.text(0, 5.2, f"{desc.upper()}", color=pitch_color, fontsize=18, fontweight='bold', ha='center')
    stats_str = f"{velo} MPH" if velo else ""
    if spin: stats_str += f" | {spin} RPM"
    ax.text(0, 4.9, stats_str, color='#FFFFFF', fontsize=12, ha='center')
    
    fig.text(0.02, 0.02, "@PitchScript", color='#FFFFFF', alpha=0.3, fontsize=16, fontweight='bold')

    sz_top = 3.5
    sz_bot = 1.5
    sz_width = 17 / 12
    sz_left = -sz_width / 2
    
    zone = patches.Rectangle((sz_left, sz_bot), sz_width, sz_top - sz_bot,
                             linewidth=3, edgecolor='#444444', facecolor='none', linestyle='dashed')
    ax.add_patch(zone)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.8, 5.5)
    ax.set_aspect('equal')
    ax.axis('off') 
    
    if sequence:
        for i, p in enumerate(sequence):
            if isinstance(p, dict) and p.get("pX") is not None and p.get("pZ") is not None:
                color = PITCH_COLORS.get(p.get("pitch_family"), "#888888")
                ax.scatter(p["pX"], p["pZ"], s=250, color=color, edgecolor='none', alpha=0.2, zorder=3)
                ax.text(p["pX"], p["pZ"], str(i+1), color='white', ha='center', va='center', fontsize=9, fontweight='bold', alpha=0.3, zorder=4)

    pX = pitch_data.get("pX")
    pZ = pitch_data.get("pZ")
    if pX is not None and pZ is not None:
        ax.scatter(pX, pZ, s=600, color=pitch_color, edgecolor='white', linewidth=2, zorder=5)
        ax.text(pX, pZ, str(len(sequence)+1), color='white', ha='center', va='center', fontsize=14, fontweight='bold', zorder=6)

def _draw_mix_donuts(fig, gs_sub, pitch_data):
    ax_mix_label = fig.add_subplot(gs_sub)
    ax_mix_label.set_facecolor('#111111')
    ax_mix_label.axis('off')
    ax_mix_label.set_title("PITCH MIXES", fontsize=14, color='#FFFFFF', fontweight='bold', pad=0)

    donut_gs = gs_sub.subgridspec(1, 3, wspace=0.1)
    
    p_last = pitch_data.get('pitcher', 'Pitcher').split(' ')[-1].upper()
    b_last = pitch_data.get('batter', 'Batter').split(' ')[-1].upper()
    balls, strikes = pitch_data.get("balls", 0), pitch_data.get("strikes", 0)
    
    mix_titles = [f"{p_last}\nOVERALL", f"{p_last}\n{balls}-{strikes}", f"VS\n{b_last}"]
    mixes = [
        (pitch_data.get("tendency_global_Fastball_pct", 0), pitch_data.get("tendency_global_Breaking_pct", 0), pitch_data.get("tendency_global_Offspeed_pct", 0)),
        (pitch_data.get("tendency_count_Fastball_pct", 0), pitch_data.get("tendency_count_Breaking_pct", 0), pitch_data.get("tendency_count_Offspeed_pct", 0)),
        (pitch_data.get("tendency_batter_count_Fastball_pct", 0), pitch_data.get("tendency_batter_count_Breaking_pct", 0), pitch_data.get("tendency_batter_count_Offspeed_pct", 0))
    ]
    
    colors = [PITCH_COLORS["Fastball"], PITCH_COLORS["Breaking"], PITCH_COLORS["Offspeed"]]
    
    for i, vals in enumerate(mixes):
        ax_donut = fig.add_subplot(donut_gs[i])
        ax_donut.set_facecolor('#111111')
        if sum(vals) == 0:
            ax_donut.pie([1], colors=['#333333'], startangle=90, wedgeprops=dict(width=0.5, edgecolor='#111111'))
            ax_donut.text(0, 0, "N/A", color='#FFFFFF', ha='center', va='center', fontsize=9)
        else:
            ax_donut.pie(vals, colors=colors, startangle=90, wedgeprops=dict(width=0.5, edgecolor='#111111'))
        
        ax_donut.set_title(mix_titles[i], color='#FFFFFF', fontsize=9, fontweight='bold', pad=5)

def _draw_prediction_bars(ax, probs, actual_fam):
    set_dark_theme(ax)
    ax.set_title("MODEL PREDICTION", fontsize=14, color='#FFFFFF', fontweight='bold', pad=10)
    
    sorted_probs = sorted(probs.items(), key=lambda x: x[1])
    p_labels = [("★ " if f == actual_fam else "") + f.upper() for f, v in sorted_probs]
    p_values = [x[1] * 100 for x in sorted_probs]
    p_colors = [PITCH_COLORS.get(f, '#FFFFFF') for f, v in sorted_probs]
    
    bars = ax.barh(p_labels, p_values, color=p_colors, height=0.5, alpha=0.8)
    ax.set_xlim(0, 100)
    for spine in ['right', 'top', 'left']:
        ax.spines[spine].set_visible(False)
    
    for tick in ax.get_yticklabels():
        if "★" in tick.get_text():
            tick.set_fontweight('bold')
            tick.set_color('#FFFFFF')
            
    ax.tick_params(left=False, labelsize=9)
    
    for bar, val in zip(bars, p_values):
        label = f"{val:.0f}%"
        ax.text(val + 3, bar.get_y() + bar.get_height()/2, label,
                va='center', color='white', fontsize=10, fontweight='bold')

def generate_pitch_infographic(pitch_data: dict, probs: dict, surprisal: float, sequence: list = None, output_path: str = "temp_plot.png") -> str:
    """
    Generates a composite infographic for a pitch.
    """
    if sequence is None:
        sequence = []
        
    fig = plt.figure(figsize=(12, 8), facecolor='#111111')
    gs = GridSpec(3, 3, height_ratios=[1, 2, 2], width_ratios=[1, 1.2, 1])

    ax_header = fig.add_subplot(gs[0, :])
    _draw_header(ax_header, pitch_data)

    ax_seq = fig.add_subplot(gs[1:, 0])
    _draw_sequence(ax_seq, sequence, pitch_data)

    ax_sz = fig.add_subplot(gs[1:, 1])
    _draw_strike_zone(ax_sz, pitch_data, sequence, fig)

    _draw_mix_donuts(fig, gs[1, 2], pitch_data)

    ax_prob = fig.add_subplot(gs[2, 2])
    _draw_prediction_bars(ax_prob, probs, pitch_data.get("pitch_family", "Unknown"))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#111111')
    plt.close()

    return output_path
