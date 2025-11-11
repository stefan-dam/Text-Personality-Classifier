import pandas as pd
import numpy as np
from data_processor import VADMapper

def process_big5_dataset():
    # Read the dataset with tab separator
    df = pd.read_csv('data/data-final.csv', sep='\t')
    
    # Extract personality trait columns
    trait_columns = {
        'EXT': [col for col in df.columns if col.startswith('EXT') and not col.endswith('E')],
        'EST': [col for col in df.columns if col.startswith('EST') and not col.endswith('E')],  # Neuroticism
        'AGR': [col for col in df.columns if col.startswith('AGR') and not col.endswith('E')],
        'CSN': [col for col in df.columns if col.startswith('CSN') and not col.endswith('E')],  # Conscientiousness
        'OPN': [col for col in df.columns if col.startswith('OPN') and not col.endswith('E')]
    }
    
    # Calculate trait scores (1-5 scale)
    trait_scores = {}
    for trait, columns in trait_columns.items():
        trait_scores[trait] = df[columns].mean(axis=1)
    
    # Convert to DataFrame
    traits_df = pd.DataFrame(trait_scores)
    
    # Rename EST to NEU and CSN to CON for consistency
    traits_df = traits_df.rename(columns={'EST': 'NEU', 'CSN': 'CON'})
    
    # Normalize to [0,1] range
    traits_df = (traits_df - 1) / 4  # Original scale is 1-5
    
    # Create VAD mapper
    mapper = VADMapper()
    
    # Print dataset statistics
    print("Dataset Statistics:")
    print("-" * 50)
    print(f"Total number of responses: {len(df)}")
    print("\nAverage trait scores (normalized 0-1):")
    print(traits_df.mean())
    
    # Convert first 5 entries to VAD scores as example
    print("\nExample Personality Profiles:")
    print("-" * 50)
    
    for i in range(5):
        big5_scores = {
            trait: traits_df.iloc[i][trait]
            for trait in traits_df.columns
        }
        
        vad = mapper.big5_to_vad(big5_scores)
        
        print(f"\nPerson {i+1}:")
        print("Big-5 scores (normalized 0-1):")
        for trait, score in big5_scores.items():
            print(f"{trait}: {score:.3f}")
        print("\nVAD scores:")
        print(f"Valence:   {vad[0]:.3f}")
        print(f"Arousal:   {vad[1]:.3f}")
        print(f"Dominance: {vad[2]:.3f}")
    
    return traits_df

if __name__ == '__main__':
    process_big5_dataset()
