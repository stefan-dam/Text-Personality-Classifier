from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def load_and_preprocess(self):
        """Load and preprocess the essays-big5 dataset from Hugging Face"""
        ds = load_dataset("jingjietan/essays-big5", split="train")
        texts = ds["text"]
        # Order: E, N, A, C, O -> consistent with model/train
        big5_scores = np.array([
            [float(row["E"]), float(row["N"]), float(row["A"]), float(row["C"]), float(row["O"]) ]
            for row in ds
        ])
        # Normalize to [0,1]
        big5_scores = (big5_scores - 1) / 4
        X_train, X_test, y_train, y_test = train_test_split(
            texts, big5_scores, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
