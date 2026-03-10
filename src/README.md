# 🛠️ Source Code Documentation (`src/`)

This directory contains the core logic for the MLB Pitch Bot, including data collection, feature engineering, model training, and the live tracking engine.

## 📁 File Descriptions

| File | Description |
| :--- | :--- |
| **`live_game_tracker.py`** | The main real-time orchestrator. Polls live MLB games, identifies interesting outcomes (strikeouts and hard hits), and coordinates inference and tweeting. |
| **`inference.py`** | A wrapper for the trained model. Handles feature preprocessing, real-time batter lookups, and calculates **Surprisal (Bits)** using Information Theory. |
| **`features.py`** | The "Feature Engine". Logic for extracting raw pitch data and computing situational context (count tendencies, leverage, handedness, and runners on base). |
| **`dataset_generator.py`** | Historical data ingestion script. Scrapes game data for a given date range and hydrates the SQLite database. |
| **`train_model.py`** | The training pipeline. Handles feature encoding, data scaling, and training the Logistic Regression model on a categorical "Pitch Family" target. |
| **`build_baseline_tendencies.py`** | Pre-calculates global and count-specific pitcher tendencies from the database to create a fast lookup "baseline" for the live tracker. |
| **`batter_tendency_processing.py`** | Logic for fetching batter stats (OBP, K%) from the MLB API and calculating advanced metrics like Whiff and Chase rates. |
| **`database.py`** | Centralized interaction layer for the SQLite database (`data/pitches.db`). |
| **`bot.py`** | Integration with the Twitter/X API (via `tweepy`) and logic for formatting engaging tactical narratives. |
| **`constants.py`** | Project-wide configuration, including pitch code groupings, surprisal thresholds, and absolute file paths. |

---

## 🔄 Application Flow

The bot operates in a logical sequence from data collection to live deployment:

### 1. Data Foundation
*   **`dataset_generator.py`**: Ingest several weeks/months of pitch data into `data/pitches.db`.
*   **`build_baseline_tendencies.py`**: Processes that database to create a "memory" of pitcher habits (`models/baseline_tendencies.pkl`).

### 2. Brain Development
*   **`train_model.py`**: Trains the "brain" using the historical data. This produces the model artifacts and encoders used for real-time predictions.

### 3. Live Operation
*   **`live_game_tracker.py`**: 
    1.  Polls live game status.
    2.  Extracts new pitches.
    3.  Passes data to **`inference.py`** for surprisal calculation.
    4.  Triggers **`bot.py`** to format and post (or simulate) a tweet if the surprisal threshold is met.

---

## 🚀 Execution Note
All scripts should be run from the project root using `uv run python -m src.<filename>`.
