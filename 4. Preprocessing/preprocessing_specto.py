import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def wav_to_spectrogram_librosa_with_plot(audio_file, file_name):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)  # sr=None keeps the original sampling rate
    
    # Generate a mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    
    # Convert the mel spectrogram to decibels for better visualization
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    
    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, cmap='viridis')
    plt.tight_layout()
    
    # Save the spectrogram as an image
    plt.savefig(file_name)
    plt.close()  

def process_audio_dir_spectro(audio_dir, spectro_dir):
    os.makedirs(spectro_dir, exist_ok=True)
    for fname in os.listdir(audio_dir):
        # now using mp3 files instead of wav
        if fname.endswith(".mp3"):
            wav_path = os.path.join(audio_dir, fname)
            spectro_path = os.path.join(spectro_dir, fname[:-4] + ".png")
            wav_to_spectrogram_librosa_with_plot(wav_path, spectro_path)

process_audio_dir_spectro("wav/dementia", "specto/ad")
process_audio_dir_spectro("wav/control", "specto/cn")