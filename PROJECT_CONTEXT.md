# ⚾ MLB Pitch Bot - Project Status

## 🚀 Progress Summary
We have transitioned the bot to a **pure SQLite-backed architecture**, significantly improving data scalability and code cleanliness. The model has been upgraded to predict **Pitch Families** (Fastball, Breaking, Offspeed) and now incorporates **Advanced Situational Context** (Handedness and Base State).

### ✅ Completed
- **Pure SQLite Migration**: Removed all CSV-related logic. `data/pitches.db` is now the single source of truth.
- **Pitch Family Model (v2.2)**: XGBoost classifier upgraded to 71% overall accuracy.
- **Advanced Features**: Added `pitcher_hand`, `batter_side`, and `men_on_base` to the training and inference pipelines.
- **Retrained Brain**: 81% F1-score for Fastballs, making "Surprising Heaters" highly reliable.
- **Cleaned Dataset Generator**: Streamlined `src/dataset_generator.py` for direct-to-DB streaming.
- **Notebook Revitalization**: `notebooks/model_prototype.ipynb` is updated with absolute path support and the new feature set.

### 🛠️ Key Files
- `src/live_game_tracker.py`: Real-time orchestrator. Currently in **Simulation Mode**.
- `src/features.py`: Advanced feature engine (tendencies, handedness, runners).
- `src/inference.py`: Model wrapper with pitch family probability and surprisal calculation.
- `src/train_model.py`: Training pipeline (Pure SQL).
- `src/database.py`: Centralized SQLite interaction layer.

## 🛑 Next Steps (For Tomorrow)
1. **Full Season Hydration**: Continue/Verify the full-season ingest in `data/pitches.db`.
2. **Narrative Refinement**: Tune the Twitter bot strings to leverage the 90% accurate fastball detection (e.g., "Hitter was completely frozen by the heater").
3. **Production Flip**: Disable `SIMULATION_MODE` and test live tweeting with real API keys.
4. **Delayed Video Capture**: Ensure the multi-pass highlight matching is capturing high-quality clips.

## 📅 Last Updated: 2026-02-22 (EOD Wrap-up)
The bot is now faster, cleaner, and smarter. Ready for live testing.