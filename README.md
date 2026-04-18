# OffPitchScript

A live MLB game analysis bot that uses machine learning to identify statistically surprising pitches and posts real-time commentary to Twitter/X.

Instead of reporting box scores, the bot acts as a tactical analyst — flagging when a pitcher throws something a batter statistically had little reason to expect, and when a hitter sits on a pitcher's tendency and punishes it.

---

## How It Works

The bot polls the MLB Stats API every 30 seconds during live games. For each pitch, it:

1. **Predicts** the probability of each pitch family (Fastball / Breaking / Offspeed) given the pitcher's tendencies, the count, the matchup, and game context
2. **Calculates surprisal** — the information-theoretic "shock value" of the actual pitch in bits: `-log₂(P(actual_pitch))`
3. **Tweets** when a surprising pitch (≥2.5 bits) results in a strikeout or hard-hit ball

### Outcome narratives

| Outcome | Trigger | Tweet label |
|---|---|---|
| Frozen | Looking strikeout on unexpected pitch | Frozen |
| Fooled | Swinging strikeout on unexpected pitch | Fooled him |
| Sitting on it | Hard hit on a high-probability pitch | Sitting on it |
| Nasty | Hard hit on a surprising pitch | Nasty |

---

## Model

**XGBoost multi-class classifier** (`multi:softprob`) predicting pitch family across three classes.

### Feature design

The dominant signal is pitcher-specific count tendencies — the frequency a given pitcher throws each pitch family at a specific count, computed with Laplace smoothing. This encodes both pitcher identity and count context in a single feature.

| Feature group | Description |
|---|---|
| `tendency_count_*_pct` | Pitcher's frequency per pitch family at this exact count |
| `tendency_global_*_pct` | Pitcher's overall repertoire usage |
| `tendency_batter_count_*_pct` | How this batter is typically attacked at this count |
| `delta_count_*` | Pitcher deviation from league average at this count |
| `is_platoon_advantage` | Same handedness matchup (switch hitters always = 1) |
| `is_leverage` | High-stakes count flag |
| `*_streak` | Consecutive same-family pitches in the at-bat |
| `pitch_count_in_game` | Pitcher fatigue proxy |

### Training

- **Data**: MLB Stats API pitch-level data, filtered to Regular Season and Postseason
- **Split**: Chronological 80/20 — no future data leaks into training
- **Sample weights**: Class-balanced (inverse frequency) × leverage weights (2× at 2-strike or 3-ball counts)
- **Baseline tendencies**: Built from training set only; Laplace-smoothed to handle sparse counts
- **Leave-one-out encoding**: Applied to tendency features during training to prevent target leakage

### Current performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Fastball | ~0.70 | ~0.73 | ~0.71 |
| Breaking | ~0.64 | ~0.65 | ~0.64 |
| Offspeed | ~0.62 | ~0.64 | ~0.63 |
| **Overall accuracy** | | | **~57%** |
| **Balanced accuracy** | | | **~67%** |

---

## Stack

| Layer | Tool |
|---|---|
| ML | XGBoost, scikit-learn |
| Data | MLB Stats API (`statsapi`), SQLite |
| Social | Twitter/X API v2 (`tweepy`) |
| Monitoring | Weights & Biases |
| Package management | `uv` |
| Scheduling | `launchd` (tracker), `cron` (retrain + monitoring) |

---

## Project Structure

```
src/
├── live_game_tracker.py      # Main polling loop and orchestrator
├── inference.py              # Model wrapper — probabilities and surprisal
├── train_model.py            # Training pipeline
├── features.py               # Feature engineering
├── baseline_manager.py       # Tendency feature hydration and fallback logic
├── build_baseline_tendencies.py  # Pre-compute pitcher tendency lookups
├── api_extractors.py         # MLB API → pitch-level rows
├── batter_tendency_processing.py # Batter whiff/chase metrics
├── bot.py                    # Tweet formatting and Twitter API
├── nightly_monitor.py        # Daily W&B health logging
├── database.py               # SQLite read/write layer
├── constants.py              # Thresholds, paths, pitch codes
└── visualization.py          # Pitch infographic generation

notebooks/
└── model_monitoring.ipynb    # Ad-hoc calibration, surprisal, and accuracy analysis

logs/
├── tracker.log               # Live tracker output
├── retrain.log               # Weekly retrain output
└── nightly.log               # Nightly monitor output
```

---

## Running Locally

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/mlb-pitch-bot.git
cd mlb-pitch-bot
uv sync
cp .env.example .env   # fill in API keys
```

### Required environment variables (`.env`)

```
TWITTER_CONSUMER_KEY=
TWITTER_CONSUMER_SECRET=
TWITTER_ACCESS_TOKEN=
TWITTER_ACCESS_TOKEN_SECRET=
WANDB_API_KEY=
WANDB_PROJECT=PitchBot
```

### Build the dataset and train

```bash
# Pull recent game data
uv run python -m src.dataset_generator --days 14

# Build pitcher tendency lookups
uv run python -m src.build_baseline_tendencies

# Train the model
uv run python -m src.train_model
```

Or run all three steps at once:

```bash
./retrain_pipeline.sh
```

### Start the live tracker

```bash
# One-off (foreground)
uv run python -m src.live_game_tracker

# Via launchd (auto-restarts on crash)
launchctl load ~/Library/LaunchAgents/com.mlb-pitch-bot.tracker.plist
launchctl unload ~/Library/LaunchAgents/com.mlb-pitch-bot.tracker.plist  # to stop
```

### Run nightly monitoring manually

```bash
uv run python -m src.nightly_monitor            # logs today
uv run python -m src.nightly_monitor --date 2026-04-14   # specific date
uv run python -m src.nightly_monitor --dry-run  # preview without pushing to W&B
```

---

## Monitoring

All training runs and nightly summaries are logged to [Weights & Biases](https://wandb.ai).

- **Training runs** (`job_type=training`) — accuracy, Brier score, per-class recall, confusion matrix, hyperparameters
- **Nightly runs** (`job_type=nightly-monitoring`) — valid prediction rate, calibration gaps, tweet rate, full per-pitch predictions table

---

## Automation (local)

| Job | Schedule | Command |
|---|---|---|
| Live tracker | On demand via launchd | `launchctl load ...tracker.plist` |
| Weekly retrain | Mondays 6am | crontab |
| Nightly monitoring | Every night 1am | crontab |
