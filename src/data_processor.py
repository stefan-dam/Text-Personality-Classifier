from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    """
    Processes the essays-big5 dataset for personality classification.
    
    JUSTIFICATION FOR DESIGN:
    - Directly predicts Big-5 traits (binary labels: 0=low, 1=high) to avoid:
      1. Information loss from dimensionality reduction (5D -> 3D via VAD)
      2. Arbitrary/unvalidated mapping weights
      3. Non-linear trait interactions being linearized
      4. Cancellation effects when multiple traits are high
    - Big-5 is the gold standard in personality psychology with decades of validation
    - Multi-label binary classification: each person can be high (1) or low (0) on each trait independently
    - Allows model to learn natural text-personality relationships without artificial intermediate representation
    """
    
    def __init__(self):
        # Trait names for reference (OCEAN model)
        self.trait_names = ['Extraversion', 'Neuroticism', 'Agreeableness', 
                           'Conscientiousness', 'Openness']

    def load_and_preprocess(self):
        """
        Load and preprocess the essays-big5 dataset from Hugging Face.
        
        Returns:
            X_train, X_test: Essay texts
            y_train, y_test: Big-5 personality scores (binary: 0 or 1)
                             Shape: (n_samples, 5) for [E, N, A, C, O]
        
        JUSTIFICATION:
        - Binary labels kept as-is (0 = low trait, 1 = high trait):
          1. Dataset provides binary classification (not continuous scores)
          2. 0/1 encoding is standard for binary classification
          3. Neural networks handle [0,1] targets naturally with sigmoid/MSE
          4. No normalization needed - data already in optimal range
        - Order [E, N, A, C, O] matches dataset convention (OCEAN model)
        - This is a multi-label binary classification problem:
          Each trait is independently 0 or 1 (person can be high in multiple traits)
        """
        ds = load_dataset("jingjietan/essays-big5", split="train")
        texts = ds["text"]
        
        # Extract Big-5 binary labels in OCEAN order
        # Dataset columns: 'O', 'C', 'E', 'A', 'N' (but we reorder to 'E', 'N', 'A', 'C', 'O')
        # Values are binary: 0 (low on trait) or 1 (high on trait)
        big5_scores = np.array([
            [float(row["E"]), float(row["N"]), float(row["A"]), 
             float(row["C"]), float(row["O"])]
            for row in ds
        ])
        
        # NO NORMALIZATION NEEDED - data is already binary (0 or 1)
        # JUSTIFICATION: 
        # - Binary values [0, 1] are already in optimal range for neural networks
        # - Previous formula (x-1)/4 was incorrect and created negative values:
        #   * 0 -> (0-1)/4 = -0.25 (WRONG!)
        #   * 1 -> (1-1)/4 = 0.0 (WRONG!)
        # - Keeping raw 0/1 values preserves binary classification semantics
        
        # Split into train/test sets
        # JUSTIFICATION: 80/20 split is standard, random_state ensures reproducibility
        X_train, X_test, y_train, y_test = train_test_split(
            texts, big5_scores, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
