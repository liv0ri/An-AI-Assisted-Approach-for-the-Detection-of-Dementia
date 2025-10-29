import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ======== CONFIG ========
output_dir = "diagnosis"
modalities = ["wav", "specto", "trans"]
original_classes = ["control", "dementia"]  # original folder names
class_map = {"control": "cn", "dementia": "ad"}  # target folder names
test_csv_name = "task1.csv"

os.makedirs(output_dir, exist_ok=True)

def get_participant_id(filename):
    return filename.split('-')[0]

# --- collect participants ---
records = []
for cls in original_classes:
    audio_folder = os.path.join("wav", cls)
    for f in os.listdir(audio_folder):
        if f.endswith(".mp3"):
            pid = get_participant_id(f)
            records.append({"participant": pid, "class": cls, "filename": f})

df = pd.DataFrame(records)

# --- split by participant ---
trainval_ids, test_ids = train_test_split(
    df["participant"].unique(), test_size=0.2, random_state=42
)
train_ids, val_ids = train_test_split(
    trainval_ids, test_size=0.2, random_state=42
)

# --- helper to copy files for train/val ---
def copy_train_val(ids, split_name):
    split_dir = os.path.join(output_dir, split_name)
    for m in modalities:
        for target_cls in class_map.values():
            os.makedirs(os.path.join(split_dir, m, target_cls), exist_ok=True)

    subset = df[df["participant"].isin(ids)]

    for _, row in subset.iterrows():
        fname = row["filename"]
        orig_cls = row["class"]
        target_cls = class_map[orig_cls]

        for modality in modalities:
            ext = ".mp3" if modality == "wav" else (".png" if modality == "specto" else ".txt")
            src_cls = orig_cls if modality == "wav" else target_cls
            src = os.path.join(modality, src_cls, fname.replace(".mp3", ext))
            dest = os.path.join(split_dir, modality, target_cls, os.path.basename(src))
            if os.path.exists(src):
                shutil.copy(src, dest)
            else:
                print(f"⚠️ Missing file: {src}")

def copy_test(ids):
    test_dir = os.path.join(output_dir, "test-dist")
    os.makedirs(test_dir, exist_ok=True)
    csv_rows = []

    for modality in modalities:
        mod_dir = os.path.join(output_dir, f"test-distaudio") if modality == "wav" else os.path.join(output_dir, f"test-dist{modality}")
        os.makedirs(mod_dir, exist_ok=True)

        subset = df[df["participant"].isin(ids)]
        for _, row in subset.iterrows():
            fname = row["filename"]
            orig_cls = row["class"]
            label_str = "ProbableAD" if orig_cls == "dementia" else "Control"

            # CSV record only once per file
            if modality == "wav":
                csv_rows.append({"filename": fname, "label": label_str})

            ext = ".mp3" if modality == "wav" else (".png" if modality == "specto" else ".txt")
            # use correct source folder
            src_cls = orig_cls if modality == "wav" else class_map[orig_cls]
            src = os.path.join(modality, src_cls, fname.replace(".mp3", ext))
            dest = os.path.join(mod_dir, os.path.basename(src))

            if os.path.exists(src):
                shutil.copy(src, dest)
            else:
                print(f"⚠️ Missing file: {src}")

    # save CSV
    csv_path = os.path.join(test_dir, test_csv_name)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"✅ Test CSV created at: {csv_path}")

# --- perform copies ---
copy_train_val(train_ids, "train")
copy_train_val(val_ids, "val")
copy_test(test_ids)

# --- summary ---
print(f"Train participants: {len(train_ids)}")
print(f"Val participants:   {len(val_ids)}")
print(f"Test participants:  {len(test_ids)}")
