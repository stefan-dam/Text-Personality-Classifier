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
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    best_val_loss = float('inf')
    patience = 2
    wait = 0
    best_epoch = 0
    for epoch in range(num_epochs):
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
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                texts = batch['text']
                outputs = model(input_ids, attention_mask, texts)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Accuracy proxy and error (MAE across all traits)
        def agg_accuracy(preds, targets):
            return 100 - np.mean(np.abs(preds - targets)) * 100
        def agg_error(preds, targets):
            return np.mean(np.abs(preds - targets))

        # Aggregate train predictions
        train_preds = []
        train_targets = []
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                texts = batch['text']
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask, texts)
                train_preds.append(outputs.cpu().numpy())
                train_targets.append(labels.cpu().numpy())
        train_preds = np.concatenate(train_preds, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)
        train_acc = agg_accuracy(train_preds, train_targets)
        train_err = agg_error(train_preds, train_targets)

        # Aggregate val predictions
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                texts = batch['text']
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask, texts)
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_err = agg_error(val_preds, val_targets)

        print(f"Epoch {epoch+1}: Train Accuracy: {train_acc:.4f}% | Train Loss: {avg_train_loss:.4f}, Train Err: {train_err:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val Err: {val_err:.4f}")
        scheduler.step(avg_val_loss)

        if epoch+1 in [3, 4]:
            torch.save(model.state_dict(), f'weights_epoch_{epoch+1}.pt')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pt')
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

def main():
    data_processor = DataProcessor()
    X_train, X_test, y_train, y_test = data_processor.load_and_preprocess()
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = PersonalityClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_dataset = TextPersonalityDataset(X_train, y_train, tokenizer)
    test_dataset = TextPersonalityDataset(X_test, y_test, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    train_model(model, train_loader, test_loader, device)

if __name__ == '__main__':
    main()

