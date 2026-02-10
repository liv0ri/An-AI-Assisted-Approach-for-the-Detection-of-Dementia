import os
import io
import wave
import numpy as np
import cv2
import torch
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms
import whisper
import librosa
from bert_image import BertImage

MODEL_PATH = 'saved_models/best_model.pth'
ROBERTA_MODEL_NAME = "FacebookAI/roberta-base"
N_FINETUNE = 4
MAX_LENGTH = 512
PROBABILITY_THRESHOLD = 0.5
TARGET_SAMPLE_RATE = 16000
SPECTROGRAM_N_MELS = 224

whisper_model = whisper.load_model("base")

MMSE_LABEL_MAP = {
    0: 'Severe (0-9)',
    1: 'Moderate (10-18)',
    2: 'Mild (19-23)',
    3: 'Normal (24-30)'
}

class MultiTaskPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
        self.model = BertImage(nfinetune=N_FINETUNE)
        self._load_model()
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizes the tensor image with a given mean and standard deviation for each color channel. 
        ])

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)
        print(f"✅ Model weights loaded from {MODEL_PATH}")

    def _process_audio_to_spectrogram(self, audio_bytes_io):
        audio_bytes_io.seek(0)
        y, sr = librosa.load(audio_bytes_io, sr=TARGET_SAMPLE_RATE)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=SPECTROGRAM_N_MELS)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Normalize to 0-255
        S_img = ((S_db - S_db.min()) / (S_db.max() - S_db.min()) * 255).astype(np.uint8)
        S_img = cv2.merge([S_img, S_img, S_img])  
        pil_image = Image.fromarray(S_img)

        pixels_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        return pixels_tensor

    def _transcribe_audio(self, audio_bytes_io):
        audio_bytes_io.seek(0)
        with open("temp.wav", "wb") as f:
            f.write(audio_bytes_io.read())
        result = whisper_model.transcribe("temp.wav")
        os.remove("temp.wav")
        return result["text"]

    def _tokenize_transcript(self, transcript_text):
        encoded = self.tokenizer(
            transcript_text,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=MAX_LENGTH
        )
        return encoded['input_ids'].to(self.device), encoded['attention_mask'].to(self.device)

    def predict(self, audio_data, transcript_text=None):
        audio_bytes_io = io.BytesIO(audio_data)
        pixels_tensor = self._process_audio_to_spectrogram(audio_bytes_io)

        if not transcript_text:
            transcript_text = self._transcribe_audio(audio_bytes_io)

        input_ids, attn_mask = self._tokenize_transcript(transcript_text)

        with torch.no_grad():
            binary_prob, mmse_logits = self.model(pixels_tensor, input_ids, attn_mask)

            # Binary dementia
            prediction_probability = binary_prob.item()
            prediction_label = 1 if prediction_probability >= PROBABILITY_THRESHOLD else 0
            label_text = 'Dementia' if prediction_label == 1 else 'NonDementia'

            # MMSE
            mmse_probs = torch.softmax(mmse_logits, dim=1).squeeze(0)
            predicted_class = torch.argmax(mmse_logits, dim=1).item()
            predicted_text = MMSE_LABEL_MAP.get(predicted_class, 'Unknown')

        return {
            'binary_prediction_probability': round(prediction_probability, 4),
            'binary_prediction_label': prediction_label,
            'binary_label_text': label_text,
            'mmse_prediction_class': predicted_class,
            'mmse_label_text': predicted_text,
            'mmse_probabilities': {lbl: round(p.item(), 4) for lbl, p in zip(MMSE_LABEL_MAP.values(), mmse_probs)}
        }


if __name__ == '__main__':
    duration = 2
    sr = TARGET_SAMPLE_RATE
    freq = 440
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    sine_wave = (0.5*np.sin(2*np.pi*freq*t)*32767).astype(np.int16)

    dummy_audio_io = io.BytesIO()
    with wave.open(dummy_audio_io, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(sine_wave.tobytes())
    dummy_audio_io.seek(0)
    dummy_audio = dummy_audio_io.read()

    predictor = MultiTaskPredictor()

    result = predictor.predict(dummy_audio)
    print(result)
