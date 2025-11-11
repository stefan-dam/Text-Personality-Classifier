import pandas as pd
import numpy as np
from data_processor import VADMapper

def analyze_big5_dataset():
    # Read the Big-5 dataset
    df = pd.read_csv('data/data-final.csv')
    
    # Calculate average scores for each trait
    trait_columns = {
        'EXT': [f'EXT{i}' for i in range(1, 11)],
        'NEU': [f'NEU{i}' for i in range(1, 11)],
        'AGR': [f'AGR{i}' for i in range(1, 11)],
        'CON': [f'CON{i}' for i in range(1, 11)],
        'OPN': [f'OPN{i}' for i in range(1, 11)]
    }
    
    # Calculate average for each trait
    trait_scores = {}
    for trait, columns in trait_columns.items():
        trait_scores[trait] = df[columns].mean(axis=1)
    
    # Convert to DataFrame
    traits_df = pd.DataFrame(trait_scores)
    
    # Normalize to [0,1] range
    traits_df = (traits_df - traits_df.min()) / (traits_df.max() - traits_df.min())
    
    # Create VAD mapper
    mapper = VADMapper()
    
    # Convert first 5 entries to VAD scores as example
    print("Sample Big-5 to VAD conversions:")
    print("-" * 50)
    
    for i in range(5):
        big5_scores = {
            trait: traits_df.iloc[i][trait]
            for trait in traits_df.columns
        }
        
        vad = mapper.big5_to_vad(big5_scores)
        
        print(f"\nPerson {i+1}:")
        print(f"Big-5 scores:")
        for trait, score in big5_scores.items():
            print(f"{trait}: {score:.3f}")
        print(f"\nVAD vector:")
        print(f"Valence:   {vad[0]:.3f}")
        print(f"Arousal:   {vad[1]:.3f}")
        print(f"Dominance: {vad[2]:.3f}")

if __name__ == '__main__':
    analyze_big5_dataset()
