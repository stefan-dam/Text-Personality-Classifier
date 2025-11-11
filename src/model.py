import torch
import torch.nn as nn
from transformers import RobertaModel
from sentence_transformers import SentenceTransformer

class PersonalityClassifier(nn.Module):
    def __init__(self, model_name='roberta-base'):
        super(PersonalityClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.sbert = SentenceTransformer('all-mpnet-base-v2')
        
        # Freeze RoBERTa parameters
        for param in self.roberta.parameters():
            param.requires_grad = False
            
        # Combined embedding dimension (RoBERTa + sBERT)
        roberta_dim = 768  # Base model hidden size
        sbert_dim = 768    # sBERT embedding dimension
        combined_dim = roberta_dim + sbert_dim
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)  # Output dimension is 3 for VAD (Valence, Arousal, Dominance)
        )
        
    def forward(self, input_ids, attention_mask, texts):
        # Get RoBERTa embeddings
        roberta_outputs = self.roberta(input_ids, attention_mask=attention_mask)
        roberta_embeddings = roberta_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Get sBERT embeddings
        sbert_embeddings = torch.tensor(self.sbert.encode(texts)).to(input_ids.device)
        
        # Concatenate embeddings
        combined_embeddings = torch.cat([roberta_embeddings, sbert_embeddings], dim=1)
        
        # Pass through classifier
        vad_predictions = self.classifier(combined_embeddings)
        
        return vad_predictions
