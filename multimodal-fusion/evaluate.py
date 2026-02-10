import os
import io
import whisper
import torch
from transformers import AutoTokenizer
from torchvision import transforms
from bert_image import BertImage

MODEL_PATH = 'saved_models/best_model.pth'
ROBERTA_MODEL_NAME = "FacebookAI/roberta-base"
N_FINETUNE = 8
# expriment with different threshold values to optimize performance IMPOOO
PROBABILITY_THRESHOLD = 0.5

# Load Whisper once
whisper_model = whisper.load_model("base")

class DementiaEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
        self.model = BertImage(nfinetune=N_FINETUNE)
        self._load_model()
        self.model.eval().to(self.device)
        
        # Same preprocessing as training
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)
        print(f" Model weights loaded from {MODEL_PATH}")

    def _tokenize_transcript(self, transcript_text):
        encoded = self.tokenizer(
            transcript_text,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        return (encoded['input_ids'].to(self.device), encoded['attention_mask'].to(self.device))

    def predict(self, spectogram_data, transcript_data):

        input_ids, attn_mask = self._tokenize_transcript(transcript_data)
        spectogram_data = self.transform(spectogram_data).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(spectogram_data, input_ids, attn_mask)
            prob = prediction.item()
            label = 1 if prob >= PROBABILITY_THRESHOLD else 0

        return {
            'prediction_probability': round(prob, 4),
            'prediction_label': label,
            'label_text': 'Dementia' if label == 1 else 'NonDementia',
            'transcript': transcript_data
        }


# --- Example ---
if __name__ == '__main__':
    predictor = DementiaEvaluator()
    # for spectogra, transcript in zip()

    # result = predictor.predict(dummy_audio)
    # print(result)
