# ⚾ MLB Pitch Bot - Project Status

## 🚀 Progress Summary
We have developed a sophisticated situational analysis bot that identifies "surprising" MLB pitches. The bot evaluates pitches against a historical baseline of pitcher tendencies to identify moments where a pitcher fools a hitter or a hitter correctly predicts a pitch.

### ✅ Completed
- **Situational Model**: XGBoost classifier trained on count-specific pitcher tendencies. (Accuracy: ~51% without physical "leaks").
- **Baseline Hydration**: `models/baseline_tendencies.pkl` allows the bot to instantly recall a pitcher's historical habit in any count.
- **Narrative Logic**: 
    - **Frozen! 🥶**: Surprise strikeout looking.
    - **Fooled him! 🔀**: Surprise strikeout swinging.
    - **Sitting on it! 🎯**: Expected pitch hit hard.
- **X API v2 Integration**: Fully configured for posting (currently in Simulation Mode).
- **Architecture Cleanup**: Removed deprecated files (`data_loader.py`, `mlb_test.py`).

### 🛠️ Key Files
- `src/live_game_tracker.py`: Real-time orchestrator. Currently in **Simulation Mode** (printing tweets to terminal).
- `src/features.py`: Feature engineering (tendencies, situational context).
- `src/inference.py`: Calculates **Surprisal (Bits)** based on model probabilities.
- `src/train_model.py`: Training pipeline for the XGBoost brain.
- `src/build_baseline_tendencies.py`: Generates the baseline memory from historical CSVs.

## 🛑 Next Steps
1. **Model Hardening**: 
   - Ingest a full season of data (2024 or 2025) to build a more robust baseline.
   - Refine features to include "Current Game Script" (rolling game-level tendencies).
2. **Production Deployment**:
   - Flip `SIMULATION_MODE` off in `live_game_tracker.py` once narratives are finalized.
   - Ensure Twitter post-rate doesn't hit X API limits.
3. **Delayed Video Integration**:
   - Refine the 2nd-pass logic to ensure video clips generated 2-5 minutes after the play are reliably captured.

## 📅 Last Updated: 2026-02-22
The bot is now a tactical analyst rather than just a box-score reporter.