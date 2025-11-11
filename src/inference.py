import torch
from transformers import RobertaTokenizer
from model import PersonalityClassifier

class PersonalityInference:
    def __init__(self, model_path='best_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.model = PersonalityClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text):
        """
        Predict VAD values for input text
        Returns: tuple of (Valence, Arousal, Dominance) values
        """
        # Tokenize input
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, [text])
            vad_values = outputs[0].cpu().numpy()
        
        return tuple(vad_values)

def main():
    # Example usage
    inference = PersonalityInference()
    
    # Test examples
    test_sentences = [
        "I am very happy today",
        "I feel anxious about the upcoming presentation",
        "I love spending time with my friends and family",
        "I prefer to work alone in a quiet environment"
    ]
    
    for sentence in test_sentences:
        valence, arousal, dominance = inference.predict(sentence)
        print(f"\nText: {sentence}")
        print(f"VAD Values:")
        print(f"Valence (positive/negative): {valence:.3f}")
        print(f"Arousal (active/passive): {arousal:.3f}")
        print(f"Dominance (dominant/submissive): {dominance:.3f}")

if __name__ == '__main__':
    main()
