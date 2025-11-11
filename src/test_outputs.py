import pandas as pd
import numpy as np
from data_processor import DataProcessor, VADMapper

def test_outputs():
    print("1. Testing VAD Mapper with sample Big-5 scores:")
    print("-" * 50)
    
    # Create a VAD mapper
    mapper = VADMapper()
    
    # Sample Big-5 scores (normalized to 0-1)
    sample_scores = {
        'EXT': 0.8,  # High extraversion
        'NEU': 0.2,  # Low neuroticism
        'AGR': 0.7,  # High agreeableness
        'CON': 0.6,  # Moderate conscientiousness
        'OPN': 0.5   # Moderate openness
    }
    
    # Get VAD values
    vad = mapper.big5_to_vad(sample_scores)
    print(f"Input Big-5 scores:")
    for trait, score in sample_scores.items():
        print(f"{trait}: {score:.3f}")
    print("\nOutput VAD values:")
    print(f"Valence:   {vad[0]:.3f}")
    print(f"Arousal:   {vad[1]:.3f}")
    print(f"Dominance: {vad[2]:.3f}")
    
    # Try processing some example text
    print("\n2. Example text-based predictions:")
    print("-" * 50)
    
    try:
        from inference import PersonalityInference
        inference = PersonalityInference()
        
        test_sentences = [
            "I am very happy today and love spending time with friends",
            "I prefer to work alone in a quiet environment",
            "I get anxious when speaking in front of large groups",
            "I enjoy organizing events and taking charge of projects"
        ]
        
        for text in test_sentences:
            print(f"\nInput text: {text}")
            try:
                vad = inference.predict(text)
                print(f"VAD values:")
                print(f"Valence:   {vad[0]:.3f}")
                print(f"Arousal:   {vad[1]:.3f}")
                print(f"Dominance: {vad[2]:.3f}")
            except Exception as e:
                print(f"Could not make prediction: {e}")
                print("Note: Model needs to be trained first")
    
    except ImportError:
        print("Note: Inference module requires trained model")

if __name__ == '__main__':
    test_outputs()
