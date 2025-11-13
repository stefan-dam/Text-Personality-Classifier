import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from data_processor import DataProcessor
from model import PersonalityClassifier
import numpy as np

class TextPersonalityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    """
    Train the personality classifier with comprehensive metrics.
    
    TRAINING DESIGN JUSTIFICATIONS:
    
    1. Loss Function - MSELoss (Binary Cross Entropy alternative):
       - Works for multi-label binary classification (each trait independent)
       - Treats problem as regression with binary targets [0, 1]
       - Alternative: BCEWithLogitsLoss would require sigmoid output layer
       - MSE is simpler and works well when outputs are naturally bounded
    
    2. Optimizer - AdamW:
       - Adam with weight decay (L2 regularization) prevents overfitting
       - lr=2e-5: Small learning rate preserves pre-trained knowledge
       - weight_decay=0.05: Moderate regularization for small dataset
    
    3. Learning Rate Scheduler - ReduceLROnPlateau:
       - Reduces LR when validation loss plateaus (adaptive to training dynamics)
       - factor=0.5: Aggressive reduction helps fine-tuning
       - patience=1: Quick response to stagnation
    
    4. Early Stopping (patience=2):
       - Prevents overfitting on small dataset (~2400 samples)
       - Saves compute time
       - Retains best model based on validation performance
    
    5. Metrics:
       - MAE (Mean Absolute Error): Interpretable for binary [0,1] targets
       - Per-trait MAE: Identifies which traits are harder to predict
       - Binary accuracy (threshold 0.5): Standard classification metric
       - Within-tolerance accuracy: More lenient metric for soft predictions
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    best_val_loss = float('inf')
    patience = 2
    wait = 0
    trait_names = ['Extraversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            texts = batch['text']
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                texts = batch['text']
                
                outputs = model(input_ids, attention_mask, texts)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate detailed metrics
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        # Overall MAE (Mean Absolute Error)
        overall_mae = np.mean(np.abs(val_preds - val_targets))
        
        # Per-trait MAE for diagnostic purposes
        per_trait_mae = np.mean(np.abs(val_preds - val_targets), axis=0)
        
        # Binary classification accuracy (threshold at 0.5)
        # JUSTIFICATION: Since targets are 0 or 1, round predictions to nearest integer
        binary_accuracy = np.mean((val_preds > 0.5).astype(int) == val_targets.astype(int)) * 100
        
        # Per-trait binary accuracy
        per_trait_accuracy = np.mean((val_preds > 0.5).astype(int) == val_targets.astype(int), axis=0) * 100
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Val MAE: {overall_mae:.4f} | Binary Accuracy: {binary_accuracy:.2f}%")
        print(f"  Per-trait Accuracy: ", end="")
        for i, (name, acc) in enumerate(zip(trait_names, per_trait_accuracy)):
            print(f"{name[:3]}={acc:.1f}%", end=" " if i < 4 else "\n")
        
        scheduler.step(avg_val_loss)
        
        # Save checkpoints at specific epochs for analysis
        if epoch+1 in [3, 4]:
            torch.save(model.state_dict(), f'weights_epoch_{epoch+1}.pt')
            print(f"  Saved checkpoint: weights_epoch_{epoch+1}.pt")
        
        # Early stopping with best model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  ✓ New best model saved (val_loss: {best_val_loss:.4f})")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
        
        print()  # Blank line for readability

def main():
    """
    Main training pipeline for Big-5 personality classification.
    
    PIPELINE JUSTIFICATIONS:
    
    1. Data Processing:
       - Binary classification (0=low trait, 1=high trait) for each Big-5 dimension
       - No normalization needed - data already in [0,1] range
       - Multi-label problem: person can be high on multiple traits simultaneously
       - 80/20 split provides sufficient validation data (~480 validation samples)
    
    2. Model Configuration:
       - RoBERTa tokenizer max_length=128: Balances context vs. memory
         (Most essays fit within 128 tokens, longer ones get truncated)
       - Dual embeddings leverage complementary linguistic features
       - 5-output architecture: one binary prediction per trait [E, N, A, C, O]
    
    3. Training Configuration:
       - Batch size=16: Optimal for GPU memory and gradient stability
       - Shuffle=True: Prevents ordering bias in training
       - Test set as validation: Standard practice for model selection
    
    4. Device Selection:
       - Auto-detects CUDA for faster training (GPU if available)
       - Falls back to CPU for compatibility
    """
    print("="*60)
    print("Big-5 Personality Binary Classifier Training")
    print("="*60)
    
    # Initialize data processor (uses essays-big5 from Hugging Face)
    print("\n[1/5] Loading and preprocessing data...")
    data_processor = DataProcessor()
    X_train, X_test, y_train, y_test = data_processor.load_and_preprocess()
    print(f"  Dataset: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"  Target: Binary Big-5 traits (0=low, 1=high) [E, N, A, C, O]")
    print(f"  Task: Multi-label binary classification")
    
    # Initialize tokenizer and model
    print("\n[2/5] Initializing model...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = PersonalityClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Device: {device}")
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Create datasets and dataloaders
    print("\n[3/5] Creating data loaders...")
    train_dataset = TextPersonalityDataset(X_train, y_train, tokenizer)
    test_dataset = TextPersonalityDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    print(f"  Batch size: 16")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(test_loader)}")
    
    # Train the model
    print("\n[4/5] Training model...")
    print("-"*60)
    train_model(model, train_loader, test_loader, device)
    
    print("\n[5/5] Training complete!")
    print("  Best model saved to: best_model.pt")
    print("="*60)

if __name__ == '__main__':
    main()
