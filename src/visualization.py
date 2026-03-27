import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

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
        
    fig = plt.figure(figsize=(12, 10), facecolor='#111111')
    gs = GridSpec(4, 2, width_ratios=[1.2, 1], height_ratios=[1.2, 1.5, 1.2, 0.8])

    # Extract Data
    pitcher = pitch_data.get("pitcher", "Pitcher")
    batter = pitch_data.get("batter", "Batter")
    inning = pitch_data.get("inning", 1)
    half = pitch_data.get("half_inning", "Top").title()
    balls = pitch_data.get("balls", 0)
    strikes = pitch_data.get("strikes", 0)
    outs = pitch_data.get("outs", 0)
    pX = pitch_data.get("pX")
    pZ = pitch_data.get("pZ")
    actual_fam = pitch_data.get("pitch_family", "Unknown")
    desc = pitch_data.get("pitch_type_desc", actual_fam)
    velo = pitch_data.get("velocity")
    spin = pitch_data.get("spin_rate")

    # --- 1. Header & Text (Top Right) ---
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.set_facecolor('#111111')
    ax_text.axis('off')
    
    y_pos = 0.9
    ax_text.text(0.0, y_pos, f"{pitcher}", color='#FFFFFF', fontsize=24, fontweight='bold')
    y_pos -= 0.1
    ax_text.text(0.0, y_pos, f"vs {batter}", color='#AAAAAA', fontsize=20)
    
    y_pos -= 0.15
    ax_text.text(0.0, y_pos, f"{half} {inning}  |  {balls}-{strikes} Count  |  {outs} Outs", color='#CCCCCC', fontsize=14)
    
    y_pos -= 0.2
    pitch_color = PITCH_COLORS.get(actual_fam, '#FFFFFF')
    ax_text.text(0.0, y_pos, f"The Pitch: {desc} ({actual_fam})", color=pitch_color, fontsize=18, fontweight='bold')
    
    y_pos -= 0.1
    stats_str = ""
    if velo: stats_str += f"{velo} MPH"
    if spin: stats_str += f"  |  {spin} RPM"
    ax_text.text(0.0, y_pos, stats_str, color='#AAAAAA', fontsize=14)
    
    y_pos -= 0.1
    stats_str = ""
    if velo: stats_str += f"{velo} MPH"
    if spin: stats_str += f"  |  {spin} RPM"
    ax_text.text(0.0, y_pos, stats_str, color='#AAAAAA', fontsize=14)

    # --- 2. Strike Zone Plot (Left Column) ---
    ax_sz = fig.add_subplot(gs[:, 0])
    set_dark_theme(ax_sz)
    ax_sz.set_title("Pitch Location (Catcher's POV)", fontsize=16, pad=15)
    
    sz_top = 3.5
    sz_bot = 1.5
    sz_width = 17 / 12  # 17 inches standard plate width
    sz_left = -sz_width / 2
    
    # Draw strike zone
    zone = patches.Rectangle((sz_left, sz_bot), sz_width, sz_top - sz_bot,
                             linewidth=2, edgecolor='#AAAAAA', facecolor='none', linestyle='dashed')
    ax_sz.add_patch(zone)
    
    # Draw home plate
    hp_y = 0.25
    hp = patches.Polygon([(-sz_width/2, hp_y), (sz_width/2, hp_y), (sz_width/2, 0), (0, -0.25), (-sz_width/2, 0)],
                         closed=True, color='#EEEEEE', alpha=0.8)
    ax_sz.add_patch(hp)
    
    ax_sz.set_xlim(-3, 3)
    ax_sz.set_ylim(0, 5)
    ax_sz.set_aspect('equal')
    ax_sz.axis('off') # Clean look
    
    # Plot previous sequence
    if sequence:
        for p in sequence:
            if p.get("pX") is not None and p.get("pZ") is not None:
                color = PITCH_COLORS.get(p.get("pitch_family"), "#888888")
                ax_sz.scatter(p["pX"], p["pZ"], s=300, color=color, edgecolor='none', alpha=0.3, zorder=3)
                ax_sz.text(p["pX"], p["pZ"], str(p.get("pitch_number", "")), 
                           color='white', ha='center', va='center', fontsize=10, alpha=0.6, zorder=4)

    # Plot the pitch
    if pX is not None and pZ is not None:
        # Note: MLB pX is from catcher's perspective. Negative is inside to RHB.
        ax_sz.scatter(pX, pZ, s=400, color=pitch_color, edgecolor='white', linewidth=2, zorder=5)
        # Add actual pitch number to center
        current_num = len(sequence) + 1
        ax_sz.text(pX, pZ, str(current_num), color='white', ha='center', va='center', fontsize=14, fontweight='bold', zorder=6)
    else:
        ax_sz.text(0, 2.5, "Location Data\nUnavailable", color='#AAAAAA', fontsize=16, ha='center', va='center')

    # --- 2b. Sequence List (Below Strike Zone) ---
    ax_seq = fig.add_subplot(gs[3, 0])
    ax_seq.set_facecolor('#111111')
    ax_seq.axis('off')
    
    if sequence or pitch_data:
        full_seq = []
        for p in sequence:
            full_seq.append({
                "name": p.get("pitch_type_desc", p.get("pitch_type_code", "Unknown")),
                "family": p.get("pitch_family", "Unknown")
            })
        full_seq.append({
            "name": pitch_data.get("pitch_type_desc", pitch_data.get("pitch_type", "Unknown")),
            "family": pitch_data.get("pitch_family", "Unknown")
        })
        
        # Render a vertical list
        start_y = 0.8
        for i, p_info in enumerate(full_seq):
            y = start_y - (i * 0.15)
            # Stop if we run out of vertical space
            if y < 0.1: break
            
            p_name = p_info["name"]
            p_fam = p_info["family"]
            color = PITCH_COLORS.get(p_fam, '#CCCCCC')
            
            ax_seq.text(0.5, y, f"{i+1}. {p_name}", color=color, fontsize=12, 
                        ha='center', va='center', fontweight='bold')

    # --- 3. Contextual Stacked Bar Charts (Middle Right) ---
    ax_ctx = fig.add_subplot(gs[1, 1])
    set_dark_theme(ax_ctx)
    ax_ctx.set_title("Historical Pitch Mixes", fontsize=14, pad=10)
    
    def plot_stacked_bar(ax, y_pos, data_dict, label, height=0.6):
        left = 0
        for fam in ["Fastball", "Breaking", "Offspeed"]:
            val = data_dict.get(fam, 0)
            if val > 0:
                color = PITCH_COLORS.get(fam, '#FFF')
                ax.barh(y_pos, val * 100, left=left * 100, height=height, color=color, edgecolor='#111111', linewidth=1)
                if val > 0.08: # Only label if > 8%
                    ax.text(left*100 + (val*100)/2, y_pos, f"{val*100:.0f}%", 
                            color='white', ha='center', va='center', fontsize=10, fontweight='bold')
                left += val
        ax.text(-3, y_pos, label, color='#CCCCCC', ha='right', va='center', fontsize=12)

    p_global = {
        "Fastball": pitch_data.get("tendency_global_Fastball_pct", 0),
        "Breaking": pitch_data.get("tendency_global_Breaking_pct", 0),
        "Offspeed": pitch_data.get("tendency_global_Offspeed_pct", 0)
    }
    p_count = {
        "Fastball": pitch_data.get("tendency_count_Fastball_pct", 0),
        "Breaking": pitch_data.get("tendency_count_Breaking_pct", 0),
        "Offspeed": pitch_data.get("tendency_count_Offspeed_pct", 0)
    }
    b_count = {
        "Fastball": pitch_data.get("tendency_batter_count_Fastball_pct", 0),
        "Breaking": pitch_data.get("tendency_batter_count_Breaking_pct", 0),
        "Offspeed": pitch_data.get("tendency_batter_count_Offspeed_pct", 0)
    }
    
    # If a mix is completely empty (e.g. 0 pitches recorded), default to empty
    if sum(p_global.values()) == 0: p_global = {"Fastball": 1.0} # Fallback visual
    
    plot_stacked_bar(ax_ctx, 2, p_global, f"{pitcher}\nOverall")
    plot_stacked_bar(ax_ctx, 1, p_count, f"{pitcher}\nin {balls}-{strikes}")
    
    if sum(b_count.values()) > 0:
        plot_stacked_bar(ax_ctx, 0, b_count, f"League vs {batter}\nin {balls}-{strikes}")
    else:
        ax_ctx.text(50, 0, "Batter Data Unavailable", color='#666', ha='center', va='center', fontsize=10, style='italic')
        ax_ctx.text(-3, 0, f"League vs {batter}\nin {balls}-{strikes}", color='#666', ha='right', va='center', fontsize=12)
        
    ax_ctx.set_xlim(0, 100)
    ax_ctx.set_ylim(-0.5, 2.8)
    ax_ctx.axis('off')

    # --- 4. Probabilities (Bottom Right) ---
    ax_prob = fig.add_subplot(gs[2, 1])
    set_dark_theme(ax_prob)
    ax_prob.set_title(f"Model Predictions (Surprisal: {surprisal:.2f})", fontsize=14, pad=10)
    
    # Sort for bar chart (highest to lowest)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1])
    families = []
    labels = []
    for f, v in sorted_probs:
        families.append(f)
        label = f
        if f == actual_fam:
            label += " (ACTUAL)"
        labels.append(label)
        
    values = [x[1] * 100 for x in sorted_probs]
    bar_colors = [PITCH_COLORS.get(f, '#FFFFFF') for f in families]
    
    bars = ax_prob.barh(labels, values, color=bar_colors, height=0.6, alpha=0.9)
    ax_prob.set_xlim(0, 100)
    ax_prob.spines['right'].set_visible(False)
    ax_prob.spines['top'].set_visible(False)
    ax_prob.spines['left'].set_visible(False)
    ax_prob.tick_params(left=False)
    ax_prob.set_xlabel("Probability (%)", color='#AAAAAA', fontsize=10)
    
    for bar, val in zip(bars, values):
        ax_prob.text(val + 2, bar.get_y() + bar.get_height()/2, f"{val:.1f}%", 
                     va='center', color='white', fontsize=12, fontweight='bold')

    plt.tight_layout()
    # Add an overall border/title
    fig.suptitle("MLB Pitch Bot Analysis", fontsize=12, color='#666666', x=0.5, y=0.98, fontweight='bold')
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#111111')
    plt.close()

    return output_path
