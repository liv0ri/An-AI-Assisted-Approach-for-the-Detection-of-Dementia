import os
import pickle
import pylangacq
import numpy as np
from random import shuffle
import soundfile as sf
from config import INPUT_BASE_PATH, PROCESSED_DATA_PATH, TARGET_AUDIO_LENGTH
# import nltk
from nltk.corpus import wordnet as wn
import random

# nltk.download("wordnet")
# nltk.download("omw-1.4")

INPUT_FOLDERS = {
    "control": os.path.join(INPUT_BASE_PATH, "cha_files", "control"),
    "dementia": os.path.join(INPUT_BASE_PATH, "cha_files", "dementia"),
}

AUDIO_FOLDERS = {
    "control": os.path.join(INPUT_BASE_PATH, "wav", "control"),
    "dementia": os.path.join(INPUT_BASE_PATH, "wav", "dementia"),
}

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name().lower() != word.lower():
                synonyms.add(lemma.name().replace("_", " "))
    return list(synonyms)

def synonym_augment(utt):
    augmented_data = []
    words = utt["words"]

    for i, w in enumerate(words):
        synonyms = get_synonyms(w)
        if not synonyms:
            continue

        new_word = random.choice(synonyms)
        new_words = words.copy()
        new_words[i] = new_word

        new_utt = utt.copy()
        new_utt["words"] = new_words
        new_utt["source_type"] = "text_augmented"
        augmented_data.append(new_utt)

    return augmented_data

def approximate_word_times(words, utterance_times):
    start, end = utterance_times
    n_words = len(words)
    step = (end - start) / n_words
    return [[start + i * step, start + (i + 1) * step] for i in range(n_words)]

def extract_utterances(cha_path, label):
    reader = pylangacq.read_chat(cha_path)
    data_points = []
    for utt in reader.utterances():
        words = [tok.word for tok in utt.tokens if tok.word and tok.word.isalpha()]
        if not utt.time_marks or not words:
            continue
        start, end = utt.time_marks

        if start is None or end is None or end < start or end - start < 1:
            continue

        # Approximate word-level timestamps
        word_times = approximate_word_times(words, (start, end))

        data_points.append({
            "words": words,
            "word_times": word_times,  
            "start": start,         
            "end": end,           
            "label": label,
            "source_file": os.path.basename(cha_path)
        })
    return data_points

def load_and_save_data(input_folders, audio_folders, output_path, target_length=TARGET_AUDIO_LENGTH):
    all_data_points = []
    augmented_points = 0
    original_points = 0
    for label, folder in input_folders.items():
        for fname in os.listdir(folder):
            if fname.endswith(".cha"):
                cha_path = os.path.join(folder, fname)
                base_name = os.path.splitext(fname)[0]
                audio_path = os.path.join(audio_folders[label], base_name + ".mp3")
                
                try:
                    audio, sample_rate = sf.read(audio_path)
                except FileNotFoundError:
                    raise FileNotFoundError(f"Audio file not found for {cha_path}, skipping.")
                
                # Check for multiple channels and convert to mono if necessary
                if audio.ndim > 1:
                    audio = audio[:, 0]
                
                utterances = extract_utterances(cha_path, label)
                
                for utt in utterances:
                    original_points += 1
                    start_frame = int((utt['start'] / 1000) * sample_rate)
                    end_frame = int((utt['end'] / 1000) * sample_rate)

                    # Ensure valid frame indices
                    start_frame = max(0, start_frame)
                    end_frame = min(len(audio), end_frame)

                    if start_frame >= end_frame:
                        continue

                    utt_audio = audio[start_frame:end_frame]

                    if len(utt_audio) > target_length:
                        utt_audio = utt_audio[:target_length]
                    else:
                        pad_len = target_length - len(utt_audio)
                        left_pad = pad_len // 2
                        right_pad = pad_len - left_pad
                        utt_audio = np.pad(utt_audio, (left_pad, right_pad))
                    
                    utt['audio'] = utt_audio.astype(np.float32)
                    utt['source_type'] = 'original'
                    all_data_points.append(utt.copy())

                    augmented_utts = synonym_augment(utt)
                    augmented_points += len(augmented_utts)
                    all_data_points.extend(augmented_utts)

    print(f"Total datapoints (original + augmented): {len(all_data_points)}")
    print(f"Original datapoints: {original_points}, Augmented datapoints: {augmented_points}")
    shuffle(all_data_points)

    with open(output_path, "wb") as f:
        pickle.dump(all_data_points, f)

if __name__ == "__main__":
    load_and_save_data(INPUT_FOLDERS, AUDIO_FOLDERS, PROCESSED_DATA_PATH)