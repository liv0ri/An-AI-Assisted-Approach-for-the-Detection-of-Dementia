import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from config import OUTPUT_DIR, MODALITIES, ORIGINAL_CLASSES, CLASS_MAP, TEST_CSV_NAME

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_participant_id(filename):
    return filename.split('-')[0]

# collect participants 
records = []
for cls in ORIGINAL_CLASSES:
    audio_folder = os.path.join(f"{MODALITIES[0]}", cls)
    for f in os.listdir(audio_folder):
        if f.endswith(".mp3"):
            pid = get_participant_id(f)
            records.append({"participant": pid, "class": cls, "filename": f})

df = pd.DataFrame(records)
print(f"Total audio files found: {len(df)}")

# split by participant
trainval_ids, test_ids = train_test_split(df["participant"].unique(), test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(trainval_ids, test_size=0.2, random_state=42)

# copy train/val 
def copy_train_val(ids, split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    for m in MODALITIES:
        for target_cls in CLASS_MAP.values():
            dest_mod = "audio" if m == f"{MODALITIES[0]}" else m  
            os.makedirs(os.path.join(split_dir, dest_mod, target_cls), exist_ok=True)

    subset = df[df["participant"].isin(ids)]
    for _, row in subset.iterrows():
        fname = row["filename"]
        orig_cls = row["class"]
        target_cls = CLASS_MAP[orig_cls]

        for modality in MODALITIES:
            ext = ".mp3" if modality == f"{MODALITIES[0]}" else (".png" if modality == "specto" else ".txt")
            src_cls = orig_cls if modality == f"{MODALITIES[0]}" else target_cls
            src = os.path.join(modality, src_cls, fname.replace(".mp3", ext))

            dest_mod = "audio" if modality == f"{MODALITIES[0]}" else modality
            dest = os.path.join(split_dir, dest_mod, target_cls, os.path.basename(src))
            if os.path.exists(src):
                shutil.copy(src, dest)
            else:
                print(f"Missing file: {src}")

# copy test together and create CSV
def copy_test(ids):
    test_dir = os.path.join(OUTPUT_DIR, "test-dist")
    os.makedirs(test_dir, exist_ok=True)
    csv_rows = []

    for modality in MODALITIES:
        mod_dir = os.path.join(OUTPUT_DIR, f"test-distaudio") if modality == f"{MODALITIES[0]}" else os.path.join(OUTPUT_DIR, f"test-dist{modality}")
        os.makedirs(mod_dir, exist_ok=True)

        subset = df[df["participant"].isin(ids)]
        for _, row in subset.iterrows():
            fname = row["filename"]
            orig_cls = row["class"]
            label_str = "ProbableAD" if orig_cls == f"{ORIGINAL_CLASSES[1]}" else "Control" 

            # CSV record only once per file 
            if modality == f"{MODALITIES[0]}":
                csv_rows.append({"filename": fname[:-4], "label": label_str})

            ext = ".mp3" if modality == f"{MODALITIES[0]}" else (".png" if modality == "specto" else ".txt")
            src_cls = orig_cls if modality == f"{MODALITIES[0]}" else CLASS_MAP[orig_cls]
            src = os.path.join(modality, src_cls, fname.replace(".mp3", ext))
            dest = os.path.join(mod_dir, os.path.basename(src))

            if os.path.exists(src):
                shutil.copy(src, dest)
            else:
                print(f" Missing file: {src}")

    # save CSV in test-dist
    csv_path = os.path.join(test_dir, TEST_CSV_NAME)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f" Test CSV created at: {csv_path}")

copy_train_val(train_ids, "train")
copy_train_val(val_ids, "val")
copy_test(test_ids)

print(f"Train participants: {len(train_ids)}")
print(f"Val participants:   {len(val_ids)}")
print(f"Test participants:  {len(test_ids)}")
