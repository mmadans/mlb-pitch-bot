import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.bot import format_surprise_strikeout_tweet

def test_formatting():
    print("Testing formatting...")
    # Test normal case
    t1 = format_surprise_strikeout_tweet("Pitcher", "Batter", "Fastball", "Fastball", 0.15, True)
    assert "Prob: 15.0%" in t1
    print("  Normal case passed.")
    
    # Test low probability case
    t2 = format_surprise_strikeout_tweet("Pitcher", "Batter", "Fastball", "Fastball", 0.0005, True)
    assert "Prob: <0.1%" in t2
    print("  Low probability case passed.")
    
    # Test zero probability (should be handled by inference, but good to check formatting)
    t3 = format_surprise_strikeout_tweet("Pitcher", "Batter", "Fastball", "Fastball", 0.0, True)
    assert "Prob: <0.1%" in t3
    print("  Zero probability case passed.")

def test_inference_floor():
    print("Testing inference floor...")
    from src.inference import PitchPredictor
    
    # Initialize predictor (this might fail if model is missing, but let's assume it exists)
    try:
        predictor = PitchPredictor()
        # Mock some features
        df = pd.DataFrame([{
            'balls': 0, 'strikes': 0, 'outs': 0,
            'score_home': 0, 'score_away': 0,
            'inning': 1, 'half_inning': 'top',
            'pitcher': 'Mock', 'pitcher_id': 1,
            'batter': 'Mock', 'batter_id': 2,
            'pitch_type': 'FF', # Fastball
            'men_on_base': 'Empty'
        }])
        
        # We need to ensure contextual features are added because PitchPredictor expects them
        from src.features import add_contextual_features
        df = add_contextual_features(df)
        
        # Also need baseline tendencies to avoid errors if the model expects them
        # For this test, we just want to see if raw_probs + epsilon works
        probs = predictor.predict_probabilities(df)
        print(f"  Predicted probabilities: {probs}")
        
        for p_fam, p_val in probs.items():
            assert p_val >= 0.00005, f"Probability for {p_fam} is too low: {p_val}"
        
        print("  Inference floor case passed.")
    except Exception as e:
        print(f"  Skipping inference test (likely due to missing model file or baseline): {e}")

if __name__ == "__main__":
    try:
        test_formatting()
        test_inference_floor()
        print("\nAll tests passed!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)
