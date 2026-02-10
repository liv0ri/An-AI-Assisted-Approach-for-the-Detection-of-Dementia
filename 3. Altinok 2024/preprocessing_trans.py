import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import string
import os

model_name = "openai/whisper-base"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

device = "cuda:0"
model.to(device)

def create_transcript(wav_path, save_path, chunk_length_s=30):
    # Load and resample audio to 16kHz
    audio, sr = librosa.load(wav_path, sr=16000)
    total_duration = librosa.get_duration(y=audio, sr=sr)

    transcripts = []

    start = 0
    while start < total_duration:
        end = min(start + chunk_length_s, total_duration)
        chunk = audio[int(start*sr):int(end*sr)]

        # Preprocess for Whisper
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt").to(device)

        # Generate prediction
        with torch.no_grad():
            predicted_ids = model.generate(
                inputs["input_features"]
            )

        # Decode transcription
        transcription = processor.decode(predicted_ids[0])
        transcription = transcription.lower()
        transcription = transcription.translate(str.maketrans('', '', string.punctuation))

        transcripts.append(transcription)

        # Adding a small gap 
        start += (chunk_length_s + 2)

    # Join all chunk transcripts
    full_transcript = " ".join(transcripts)

    # Save to file
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(full_transcript)

def process_audio_dir_transcript(audio_dir, trans_dir):
    os.makedirs(trans_dir, exist_ok=True)
    for fname in os.listdir(audio_dir):
        if fname.endswith(".wav"):
            wav_path = os.path.join(audio_dir, fname)
            trans_path = os.path.join(trans_dir, fname[:-4] + ".txt")
            create_transcript(wav_path, trans_path)


# Example usage
process_audio_dir_transcript("diagnosis/train/audio/ad", "diagnosis/train/trans/ad")
process_audio_dir_transcript("diagnosis/train/audio/cn", "diagnosis/train/trans/cn")
process_audio_dir_transcript("diagnosis/test-distaudio", "diagnosis/test-disttrans")