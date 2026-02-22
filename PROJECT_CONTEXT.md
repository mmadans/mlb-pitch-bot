# ⚾ MLB Pitch Bot - Current Status & Handoff

## 🚀 Progress Summary
We have successfully built the core pipeline for identifies "surprising" MLB pitches and posting them to Twitter based on specific game outcomes.

### ✅ Completed
- **Model Training**: Trained an XGBoost classifier on historical data. Models saved to `/models/`.
- **Outcome Filtering**: Implemented logic to only analyze pitches resulting in **Strikeouts** or **Hard Hits (98+ mph)** that are not outs.
- **X API v2 Integration**: Switched the bot to use the modern API v2 via `tweepy.Client`.
- **Feature Traceability**: Added `at_bat_index` and `pitch_index` to all feature engineering steps to ensure live data matches historical tendencies.

### 🛠️ Key Files
- `src/live_game_tracker.py`: The "Brain". Monitors live games and triggers the analysis/tweeting.
- `src/bot.py`: The "Voice". Handles X API v2 authentication and tweet formatting.
- `src/train_model.py`: Used to update the model.

## 🛑 Current Blocker / Next Steps
1. **Twitter Authentication Fix**:
   - We hit a `403 Forbidden` error during the test post.
   - **Action needed**: Go to the X Developer Portal, change App Permissions to **"Read and Write"**, and then **REGENERATE** the Access Token and Access Token Secret. Update `.env` with these new tokens.
2. **First Live Test**:
   - Run `uv run python -m src.live_game_tracker` to start the live monitoring.
3. **Simulation (Optional)**:
   - Create a script to simulate a past game to verify the tweet formatting and surprisal calculation without waiting for a live event.

## 📅 Last Saved: 2026-02-22 13:10
The bot is primed and the code is fully synchronized with the current MLB season data. Enjoy your break!