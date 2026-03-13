#!/bin/bash
# MLB Pitch Bot - Automated Retraining Pipeline
# This script updates the pitch dataset and retrains the model.

echo "--- Starting MLB Pitch Bot Retraining Pipeline ---"
DATE=$(date +"%Y-%m-%d %H:%M:%S")
echo "Time: $DATE"

# 1. Update dataset (Pull last 14 days to ensure recent trends are captured)
echo "Step 1: Scrapping recent games..."
uv run python -m src.dataset_generator --days 14

# 2. Update Baseline Tendencies (Required for inference)
echo "Step 2: Updating baseline tendencies..."
uv run python -m src.build_baseline_tendencies

# 3. Retrain the model (Uses the last 60 days of data from DB)
echo "Step 3: Retraining the model..."
uv run python -m src.train_model

echo "--- Pipeline Complete! New model and artifacts are ready. ---"
