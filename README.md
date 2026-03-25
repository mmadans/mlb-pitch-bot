# ⚾ MLB Pitch Bot

A sophisticated MLB pitch analysis bot that uses machine learning and historical pitcher tendencies to identify "surprising" moments in live games. Instead of just reporting box scores, this bot acts as a tactical analyst, flagging when a pitcher fools a hitter or when a hitter correctly predicts a pitcher's habit.

## 🚀 Core Features

- **Situational Surprisal**: Calculates the "shock value" of a pitch in **bits** (Information Theory) based on count-specific historical tendencies.
- **Narrative Intelligence**: Automatically categorizes outcomes into engaging tactical stories:
    - **Frozen! 🥶**: When a pitcher catches a hitter looking for a strikeout with a pitch they statistically didn't expect.
    - **Fooled him! 🔀**: When a hitter strikes out swinging on a pitch they weren't expecting.
    - **Sitting on it! 🎯**: When a hitter punishes a high-probability pitch, showing they were "waiting" for the pitcher's tendency.
- **Live Tracking**: Polls MLB StatsAPI every 30 seconds to catch interesting strikeouts and hard hits (98+ mph) as they happen.
- **Video Integration**: Automatically fetches and attaches official MLB video highlights to tweets (usually available 2-5 minutes after the play).

## 📊 Model Features & Importances

The prediction engine relies on a Calibrated XGBoost Classifier that heavily weighs historical situational tendencies over raw pitch data. The most predictive features (by weight) are:

1. **Pitcher's Count-Specific Tendencies (~55% combined)**: The exact frequency of a specific pitch (Changeup, Slider, Curveball, etc.) thrown by the pitcher in the exact current count (e.g., `tendency_count_CH_pct`).
2. **Batter's Profile in the Count (~17% combined)**: How the league typically attacks this specific batter in the current count (e.g., `tendency_batter_count_Fastball_pct`).
3. **Pitcher's Global Tendencies (~10% combined)**: The pitcher's overall repertoire usage regardless of count (e.g., `tendency_global_EP_pct`).
4. **Contextual Matchups (~5% combined)**: Platoon advantage (`is_platoon_advantage`) and overall pitcher hand bias.

## 🛠️ Technology Stack

- **ML Backend**: XGBoost Classifier trained on situational context and pitcher habits.
- **Data Source**: MLB StatsAPI (via `statsapi`).
- **Social**: Twitter/X API v2 (via `tweepy`).
- **Dev Ops**: Managed with `uv` for fast, reproducible Python environments.


## 🏃‍♂️ Running the Pipeline

The bot operates in four distinct stages:

### 1. Data Collection
Generate a historical dataset (e.g., from the last 30 days) to train the model.
```bash
uv run python -m src.dataset_generator --start 2024-05-01 --end 2024-05-30
```

### 2. Build Memory (Baseline)
Pre-calculate pitcher tendencies for fast live lookup.
```bash
uv run python -m src.build_baseline_tendencies
```



### 4. Start Live Tracking
Launch the bot to monitor today's games.
```bash
uv run python -m src.live_game_tracker
```
*(Note: Initial runs default to **Simulation Mode**, printing tweets to the terminal instead of posting them.)*

## 📁 Project Structure

- `src/live_game_tracker.py`: Real-time orchestrator and polling loop.
- `src/features.py`: The "Feature Engine" that builds situational rows.
- `src/inference.py`: Model wrapper that calculates **Surprisal (Bits)**.
- `src/bot.py`: Formatting logic and Twitter API integration.
- `src/constants.py`: Centralized thresholds and model paths.

## 🔮 Future Roadmap

- **Rolling Game Script**: Adjust tendencies in real-time if a pitcher is "off" their normal habits today.
- **Batter Tendencies**: Factor in how a batter struggles against specific pitch families.
- **Enhanced Visuals**: Generate dynamic overlay graphics for "frozen" moments.

---
*Built for the love of the game and the numbers behind it.* ⚾📈
