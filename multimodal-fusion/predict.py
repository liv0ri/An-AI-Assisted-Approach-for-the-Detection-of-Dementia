import os
import io
import wave
import numpy as np
import cv2
import librosa
import whisper
import torch
from transformers import AutoTokenizer
from PIL import Image        
from torchvision import transforms
from bert_image import BertImage

MODEL_PATH = 'saved_models/best_model.pth'
ROBERTA_MODEL_NAME = "FacebookAI/roberta-base"
N_FINETUNE = 4
MAX_LENGTH = 512
PROBABILITY_THRESHOLD = 0.5
TARGET_SAMPLE_RATE = 16000
SPECTROGRAM_N_MELS = 224

# Load Whisper once
whisper_model = whisper.load_model("base")

class DementiaPredictor:
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

    def _process_audio_to_spectrogram(self, audio_bytes_io):
        y, sr = librosa.load(audio_bytes_io, sr=TARGET_SAMPLE_RATE)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=SPECTROGRAM_N_MELS)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Normalize to 0–255
        S_img = ((S_dB - S_dB.min()) / (S_dB.max() - S_dB.min()) * 255).astype(np.uint8)
        S_img = cv2.merge([S_img, S_img, S_img]) 
        pil_image = Image.fromarray(S_img)

        return self.transform(pil_image).unsqueeze(0).to(self.device)

    def _transcribe_audio(self, audio_bytes_io):
        audio_bytes_io.seek(0)
        with open("temp.mp3", "wb") as f:
            f.write(audio_bytes_io.read())
        result = whisper_model.transcribe("temp.mp3")
        os.remove("temp.mp3")
        return result["text"]

    def _tokenize_transcript(self, transcript_text):
        encoded = self.tokenizer(
            transcript_text,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=MAX_LENGTH
        )
        return (encoded['input_ids'].to(self.device), encoded['attention_mask'].to(self.device))

    def predict(self, audio_data, transcript_text=None):
        audio_bytes_io = io.BytesIO(audio_data)
        pixels_tensor = self._process_audio_to_spectrogram(audio_bytes_io)

        if not transcript_text:
            transcript_text = self._transcribe_audio(audio_bytes_io)

        input_ids, attn_mask = self._tokenize_transcript(transcript_text)

        with torch.no_grad():
            prediction = self.model(pixels_tensor, input_ids, attn_mask)
            prob = prediction.item()
            label = 1 if prob >= PROBABILITY_THRESHOLD else 0

        return {
            'prediction_probability': round(prob, 4),
            'prediction_label': label,
            'label_text': 'Dementia' if label == 1 else 'NonDementia',
            'transcript': transcript_text
        }


# --- Example ---
if __name__ == '__main__':
    predictor = DementiaPredictor()

    # Generate dummy audio (440 Hz sine wave)
    def generate_dummy_wav_bytes(duration=3, sr=TARGET_SAMPLE_RATE, freq=440):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * freq * t)
        waveform_int16 = (waveform * 32767).astype(np.int16)

        byte_io = io.BytesIO()
        with wave.open(byte_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(waveform_int16.tobytes())
        byte_io.seek(0)
        return byte_io.getvalue()

    dummy_audio = generate_dummy_wav_bytes()
    result = predictor.predict(dummy_audio)
    print(result)
