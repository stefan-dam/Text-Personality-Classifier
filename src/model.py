import torch
import torch.nn as nn
from transformers import RobertaModel
from sentence_transformers import SentenceTransformer

class PersonalityClassifier(nn.Module):
    """
    Dual-embedding personality classifier for Big-5 trait prediction.
    
    ARCHITECTURE JUSTIFICATIONS:
    1. Dual embeddings (RoBERTa + sBERT):
       - RoBERTa: Captures token-level context, syntax, and linguistic patterns
       - sBERT: Captures semantic meaning and overall essay coherence
       - Complementary strengths provide richer representation than either alone
    
    2. Frozen base models:
       - Prevents overfitting on relatively small personality dataset (~2400 essays)
       - Leverages pre-trained knowledge from billions of tokens
       - Reduces trainable parameters by ~90%, faster training
       - Only the classifier head learns personality-specific patterns
    
    3. Architecture depth (1536 -> 512 -> 256 -> 5):
       - Gradual dimensionality reduction prevents information bottlenecks
       - 512 and 256 hidden layers allow learning complex trait interactions
       - Dropout (0.2) prevents overfitting while maintaining capacity
    
    4. Output dimension = 5 (Big-5 traits):
       - Direct prediction of psychologically validated constructs
       - Each output unit learns independent trait patterns
       - No forced correlation from intermediate VAD representation
    """
    
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
        
        # Classifier layers for Big-5 personality prediction
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)  # Output: 5 Big-5 traits [E, N, A, C, O]
        )
        
    def forward(self, input_ids, attention_mask, texts):
        """
        Forward pass combining RoBERTa and sBERT embeddings.
        
        Args:
            input_ids: Tokenized input for RoBERTa
            attention_mask: Attention mask for RoBERTa
            texts: Raw text strings for sBERT
            
        Returns:
            big5_predictions: Tensor of shape (batch_size, 5)
                             Predicted scores for [E, N, A, C, O]
        """
        # Get RoBERTa embeddings using [CLS] token
        # JUSTIFICATION: [CLS] token aggregates sequence information in BERT-style models
        roberta_outputs = self.roberta(input_ids, attention_mask=attention_mask)
        roberta_embeddings = roberta_outputs.last_hidden_state[:, 0, :]
        
        # Get sBERT embeddings
        # JUSTIFICATION: sBERT provides sentence-level semantic representation
        sbert_embeddings = torch.tensor(self.sbert.encode(texts)).to(input_ids.device)
        
        # Concatenate embeddings
        # JUSTIFICATION: Concatenation preserves all information from both models
        # Alternative (averaging) would lose model-specific patterns
        combined_embeddings = torch.cat([roberta_embeddings, sbert_embeddings], dim=1)
        
        # Pass through classifier to predict Big-5 traits
        big5_predictions = self.classifier(combined_embeddings)
        
        return big5_predictions
