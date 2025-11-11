import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_processor import VADMapper

def download_dataset():
    """Download the Big-5 personality dataset from Kaggle"""
    try:
        os.system('kaggle datasets download -d tunguz/big-five-personality-test -p data/')
        os.system('cd data && unzip big-five-personality-test.zip')
        print("Dataset downloaded successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {e}")

def process_big5_dataset():
    """Process the Big-5 personality dataset"""
    # Read the data
    data_path = 'data/data-final.csv'
    df = pd.read_csv(data_path)
    
    # List of questions for each trait
    trait_questions = {
        'EXT': [f'EXT{i}' for i in range(1, 11)],  # 10 questions
        'NEU': [f'NEU{i}' for i in range(1, 11)],  # 10 questions
        'AGR': [f'AGR{i}' for i in range(1, 11)],  # 10 questions
        'CON': [f'CON{i}' for i in range(1, 11)],  # 10 questions
        'OPN': [f'OPN{i}' for i in range(1, 11)]   # 10 questions
    }
    
    # Calculate average scores for each trait
    trait_scores = pd.DataFrame()
    for trait, questions in trait_questions.items():
        trait_scores[trait] = df[questions].mean(axis=1)
    
    # Normalize scores to [0, 1] range
    scaler = StandardScaler()
    normalized_scores = scaler.fit_transform(trait_scores)
    normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min())
    trait_scores = pd.DataFrame(normalized_scores, columns=trait_scores.columns)
    
    # Create text descriptions from the scores
    def create_text_description(row):
        traits = []
        if row['EXT'] > 0.6: traits.append("outgoing and sociable")
        elif row['EXT'] < 0.4: traits.append("reserved and introspective")
        
        if row['NEU'] > 0.6: traits.append("sensitive to stress")
        elif row['NEU'] < 0.4: traits.append("emotionally stable")
        
        if row['AGR'] > 0.6: traits.append("cooperative and compassionate")
        elif row['AGR'] < 0.4: traits.append("direct and straightforward")
        
        if row['CON'] > 0.6: traits.append("organized and responsible")
        elif row['CON'] < 0.4: traits.append("flexible and spontaneous")
        
        if row['OPN'] > 0.6: traits.append("curious and creative")
        elif row['OPN'] < 0.4: traits.append("practical and conventional")
        
        return "I am " + ", ".join(traits) if traits else "I have balanced personality traits"
    
    # Create text descriptions
    texts = trait_scores.apply(create_text_description, axis=1)
    
    # Convert to VAD vectors
    vad_mapper = VADMapper()
    vad_vectors = []
    
    for _, row in trait_scores.iterrows():
        big5_scores = {trait: row[trait] for trait in trait_scores.columns}
        vad_vectors.append(vad_mapper.big5_to_vad(big5_scores))
    
    vad_vectors = np.array(vad_vectors)
    
    # Create final dataset
    final_df = pd.DataFrame({
        'text': texts,
        'EXT': trait_scores['EXT'],
        'NEU': trait_scores['NEU'],
        'AGR': trait_scores['AGR'],
        'CON': trait_scores['CON'],
        'OPN': trait_scores['OPN'],
        'valence': vad_vectors[:, 0],
        'arousal': vad_vectors[:, 1],
        'dominance': vad_vectors[:, 2]
    })
    
    # Save processed dataset
    final_df.to_csv('data/processed_big5.csv', index=False)
    print(f"Dataset processed and saved! Shape: {final_df.shape}")
    
    # Print sample
    print("\nSample entries:")
    print(final_df[['text', 'valence', 'arousal', 'dominance']].head())

if __name__ == '__main__':
    if not os.path.exists('data/data-final.csv'):
        download_dataset()
    process_big5_dataset()
