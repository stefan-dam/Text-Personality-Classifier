from data_processor import DataProcessor, VADMapper
import pandas as pd

def test_vad_mapping():
    # Create a VADMapper instance
    mapper = VADMapper()
    
    # Test a sample personality profile
    big5_scores = {
        'EXT': 0.8,  # High extraversion
        'NEU': 0.2,  # Low neuroticism
        'AGR': 0.7,  # High agreeableness
        'CON': 0.6,  # Moderate conscientiousness
        'OPN': 0.5   # Moderate openness
    }
    
    vad = mapper.big5_to_vad(big5_scores)
    print("Sample Big-5 to VAD conversion:")
    print(f"Big-5 scores: {big5_scores}")
    print(f"VAD vector: Valence={vad[0]:.3f}, Arousal={vad[1]:.3f}, Dominance={vad[2]:.3f}")
    
    # Test with the sample dataset
    processor = DataProcessor('data/test_sample.csv')
    X_train, X_test, y_train, y_test = processor.load_and_preprocess()
    
    print("\nDataset Processing Results:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Print example text and its corresponding VAD values
    print("\nExample from processed dataset:")
    print(f"Text: {X_train[0]}")
    print(f"VAD values: {y_train[0]}")

if __name__ == '__main__':
    test_vad_mapping()
