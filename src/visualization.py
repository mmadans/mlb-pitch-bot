import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from src.utils import get_pitch_abbr

PITCH_COLORS = {
    "Fastball": "#FF4136",  # Red
    "Breaking": "#0074D9",   # Blue
    "Offspeed": "#2ECC40"    # Green
}

def set_dark_theme(ax):
    """Applies a premium dark theme to a matplotlib axis."""
    ax.set_facecolor('#111111')
    ax.tick_params(colors='#AAAAAA')
    for spine in ax.spines.values():
        spine.set_color('#333333')
    ax.xaxis.label.set_color('#AAAAAA')
    ax.yaxis.label.set_color('#AAAAAA')
    ax.title.set_color('#FFFFFF')

def generate_pitch_infographic(pitch_data: dict, probs: dict, surprisal: float, sequence: list = None, output_path: str = "temp_plot.png") -> str:
    """
    Generates a composite infographic for a pitch.
    """
    if sequence is None:
        sequence = []
        
    fig = plt.figure(figsize=(12, 8), facecolor='#111111')
    gs = GridSpec(3, 3, height_ratios=[1, 2, 2], width_ratios=[1, 1.2, 1])

    # Extract Data
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
    
    # Score and Team Info
    a_team = pitch_data.get("away_team", "AWY")
    h_team = pitch_data.get("home_team", "HOM")
    a_score = pitch_data.get("score_away", 0)
    h_score = pitch_data.get("score_home", 0)
    score_str = f"{a_team} {a_score}, {h_team} {h_score}"
    
    # Runners info
    runners = pitch_data.get("men_on_base", "Empty")
    runners_str = f"Bases: {runners.replace('_', ' ')}"
    
    pX = pitch_data.get("pX")
    pZ = pitch_data.get("pZ")
    actual_fam = pitch_data.get("pitch_family", "Unknown")
    desc = pitch_data.get("pitch_type_desc", actual_fam)
    velo = pitch_data.get("velocity")
    spin = pitch_data.get("spin_rate")
    pitch_color = PITCH_COLORS.get(actual_fam, '#FFFFFF')

    # --- 1. Header & Branding (Top Row, Full Width) ---
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor('#111111')
    ax_header.axis('off')
    
    # NEW TOP LEFT: Matchup
    ax_header.text(0.02, 0.7, pitcher.upper(), color='#FFFFFF', fontsize=22, fontweight='bold')
    ax_header.text(0.02, 0.3, f"vs {batter.upper()}", color='#FFFFFF', fontsize=18, fontweight='bold', style='italic')
    
    # NEW TOP RIGHT: All Game Context
    ax_header.text(0.98, 0.8, f"{a_team} @ {h_team}  |  {score_str}", color='#FFFFFF', fontsize=18, fontweight='bold', ha='right')
    ax_header.text(0.98, 0.45, f"{half.capitalize()} {inning}  |  {outs} OUTS", color='#0074D9', fontsize=14, fontweight='bold', ha='right')
    ax_header.text(0.98, 0.15, f"{balls}-{strikes} COUNT  |  {runners_str.upper()}", color='#FFFFFF', fontsize=12, fontweight='bold', ha='right')
    
    # --- 2. Left Column: Pitch Sequence (Vertical List) ---
    ax_seq = fig.add_subplot(gs[1:, 0])
    ax_seq.set_facecolor('#111111')
    ax_seq.axis('off')
    ax_seq.set_title("SEQUENCE", fontsize=14, color='#FFFFFF', fontweight='bold', pad=10)
    
    if sequence or pitch_data:
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
            "family": pitch_data.get("pitch_family", "Unknown"),
            "outcome": pitch_data.get("call", "")
        })
        
        # Top align the list
        spacing = 0.12
        start_y = 0.95
        
        for i, p_info in enumerate(full_seq):
            y = start_y - (i * spacing)
            if y < 0.05: break
            
            p_name = p_info["name"]
            p_fam = p_info["family"]
            p_outcome = p_info["outcome"]
            outcome_str = f" ({p_outcome})" if p_outcome else ""
            color = PITCH_COLORS.get(p_fam, '#CCCCCC')
            
            # Draw perfect circle with number using scatter
            ax_seq.scatter([0.15], [y], s=500, color=color, alpha=0.8, transform=ax_seq.transAxes, zorder=2)
            ax_seq.text(0.15, y, str(i+1), color='white', ha='center', va='center', fontsize=9, fontweight='bold', transform=ax_seq.transAxes, zorder=3)
            
            ax_seq.text(0.25, y, f"{get_pitch_abbr(p_name)}", color=color, fontsize=12, fontweight='bold', transform=ax_seq.transAxes)
            ax_seq.text(0.45, y, f"{outcome_str}", color='#FFFFFF', fontsize=11, style='italic', transform=ax_seq.transAxes) # Bigger

    # --- 3. Middle Column: Strike Zone & Pitch Details ---
    ax_sz = fig.add_subplot(gs[1:, 1])
    set_dark_theme(ax_sz)
    
    # Specific Pitch Call
    ax_sz.text(0, 5.2, f"{desc.upper()}", color=pitch_color, fontsize=18, fontweight='bold', ha='center')
    stats_str = f"{velo} MPH" if velo else ""
    if spin: stats_str += f" | {spin} RPM"
    ax_sz.text(0, 4.9, stats_str, color='#FFFFFF', fontsize=12, ha='center')
    
    # Watermark Branding (Bottom Left)
    fig.text(0.02, 0.02, "@PitchScript", color='#FFFFFF', alpha=0.3, fontsize=16, fontweight='bold')

    sz_top = 3.5
    sz_bot = 1.5
    sz_width = 17 / 12
    sz_left = -sz_width / 2
    
    zone = patches.Rectangle((sz_left, sz_bot), sz_width, sz_top - sz_bot,
                             linewidth=3, edgecolor='#444444', facecolor='none', linestyle='dashed')
    ax_sz.add_patch(zone)
    
    ax_sz.set_xlim(-3, 3)
    ax_sz.set_ylim(-0.8, 5.5) # Increased range for battery text
    ax_sz.set_aspect('equal')
    ax_sz.axis('off') 
    
    # Plot dots
    if sequence:
        for i, p in enumerate(sequence):
            if isinstance(p, dict) and p.get("pX") is not None and p.get("pZ") is not None:
                color = PITCH_COLORS.get(p.get("pitch_family"), "#888888")
                ax_sz.scatter(p["pX"], p["pZ"], s=250, color=color, edgecolor='none', alpha=0.2, zorder=3)
                # Add faded number for tracking
                ax_sz.text(p["pX"], p["pZ"], str(i+1), color='white', ha='center', va='center', fontsize=9, fontweight='bold', alpha=0.3, zorder=4)

    if pX is not None and pZ is not None:
        ax_sz.scatter(pX, pZ, s=600, color=pitch_color, edgecolor='white', linewidth=2, zorder=5)
        ax_sz.text(pX, pZ, str(len(sequence)+1), color='white', ha='center', va='center', fontsize=14, fontweight='bold', zorder=6)

    # --- 4. Right Column: Mixes (Top) + Model Prediction (Bottom) ---
    # Overall Title for Right column
    ax_mix_label = fig.add_subplot(gs[1, 2])
    ax_mix_label.set_facecolor('#111111')
    ax_mix_label.axis('off')
    ax_mix_label.set_title("PITCH MIXES", fontsize=14, color='#FFFFFF', fontweight='bold', pad=0)

    # Arrange donuts horizontally
    donut_gs = gs[1, 2].subgridspec(1, 3, wspace=0.1) # Reduced wspace to allow bigger donuts
    
    p_name_only = pitch_data.get('pitcher', 'Pitcher')
    p_last = p_name_only.split(' ')[-1].upper()
    b_name_only = pitch_data.get('batter', 'Batter')
    b_last = b_name_only.split(' ')[-1].upper()
    
    mix_titles = [f"{p_last}\nOVERALL", f"{p_last}\n{balls}-{strikes}", f"VS\n{b_last}"]
    mixes = [
        (pitch_data.get("tendency_global_Fastball_pct", 0), pitch_data.get("tendency_global_Breaking_pct", 0), pitch_data.get("tendency_global_Offspeed_pct", 0)),
        (pitch_data.get("tendency_count_Fastball_pct", 0), pitch_data.get("tendency_count_Breaking_pct", 0), pitch_data.get("tendency_count_Offspeed_pct", 0)),
        (pitch_data.get("tendency_batter_count_Fastball_pct", 0), pitch_data.get("tendency_batter_count_Breaking_pct", 0), pitch_data.get("tendency_batter_count_Offspeed_pct", 0))
    ]
    
    labels = ["Fastball", "Breaking", "Offspeed"]
    colors = [PITCH_COLORS[l] for l in labels]
    
    for i, vals in enumerate(mixes):
        ax_donut = fig.add_subplot(donut_gs[i])
        ax_donut.set_facecolor('#111111')
        if sum(vals) == 0:
            ax_donut.pie([1], colors=['#333333'], startangle=90, wedgeprops=dict(width=0.5, edgecolor='#111111')) # Wider wedge
            ax_donut.text(0, 0, "N/A", color='#FFFFFF', ha='center', va='center', fontsize=9)
        else:
            ax_donut.pie(vals, colors=colors, startangle=90, wedgeprops=dict(width=0.5, edgecolor='#111111')) # Wider wedge
        
        ax_donut.set_title(mix_titles[i], color='#FFFFFF', fontsize=9, fontweight='bold', pad=5)

    # Prediction bar chart
    ax_prob = fig.add_subplot(gs[2, 2])
    set_dark_theme(ax_prob)
    ax_prob.set_title("MODEL PREDICTION", fontsize=14, color='#FFFFFF', fontweight='bold', pad=10)
    
    sorted_probs = sorted(probs.items(), key=lambda x: x[1])
    p_labels = [f.upper() + (" ★" if f == actual_fam else "") for f, v in sorted_probs]
    p_values = [x[1] * 100 for x in sorted_probs]
    p_colors = [PITCH_COLORS.get(f, '#FFFFFF') for f, v in sorted_probs]
    
    bars = ax_prob.barh(p_labels, p_values, color=p_colors, height=0.5, alpha=0.8)
    ax_prob.set_xlim(0, 100)
    ax_prob.spines['right'].set_visible(False)
    ax_prob.spines['top'].set_visible(False)
    ax_prob.spines['left'].set_visible(False)
    ax_prob.tick_params(left=False, labelsize=9)
    
    for bar, val in zip(bars, p_values):
        ax_prob.text(val + 3, bar.get_y() + bar.get_height()/2, f"{val:.0f}%", 
                     va='center', color='white', fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#111111')
    plt.close()

    return output_path
