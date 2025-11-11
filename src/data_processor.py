from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split

class VADMapper:
    """Maps Big-5 personality traits to VAD (Valence-Arousal-Dominance) space"""
    
    def __init__(self):
        # Define mapping weights for each Big-5 trait to VAD dimensions
        # These weights should be fine-tuned based on psychological research
        self.mapping_weights = {
            'EXT': {'V': 0.6, 'A': 0.4, 'D': 0.8},  # Extraversion
            'NEU': {'V': -0.7, 'A': 0.6, 'D': -0.4}, # Neuroticism
            'AGR': {'V': 0.5, 'A': -0.3, 'D': -0.2}, # Agreeableness
            'CON': {'V': 0.3, 'A': -0.1, 'D': 0.4},  # Conscientiousness
            'OPN': {'V': 0.2, 'A': 0.3, 'D': 0.3}    # Openness
        }
    
    def big5_to_vad(self, big5_scores):
        """Convert Big-5 scores to VAD vector"""
        vad = {
            'V': 0.0,
            'A': 0.0,
            'D': 0.0
        }
        
        # Normalize Big-5 scores and compute weighted sum for each VAD dimension
        for trait, scores in big5_scores.items():
            for dim in vad.keys():
                vad[dim] += scores * self.mapping_weights[trait][dim]
        
        # Normalize VAD scores to [-1, 1] range
        for dim in vad.keys():
            vad[dim] = np.clip(vad[dim], -1, 1)
            
        return np.array([vad['V'], vad['A'], vad['D']])

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.vad_mapper = VADMapper()

    def load_and_preprocess(self):
        """Load and preprocess the essays-big5 dataset from Hugging Face"""
        ds = load_dataset("jingjietan/essays-big5", split="train")
        texts = ds["text"]
        # Big-5 columns: 'O', 'C', 'E', 'A', 'N'
        big5_scores = np.array([
            [float(row["E"]), float(row["N"]), float(row["A"]), float(row["C"]), float(row["O"])]
            for row in ds
        ])
        # Normalize scores to [0, 1] (original scale is 1-5)
        big5_scores = (big5_scores - 1) / 4
        vad_vectors = np.array([
            self.vad_mapper.big5_to_vad({
                'EXT': scores[0],
                'NEU': scores[1],
                'AGR': scores[2],
                'CON': scores[3],
                'OPN': scores[4]
            }) for scores in big5_scores
        ])
        X_train, X_test, y_train, y_test = train_test_split(
            texts, vad_vectors, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
