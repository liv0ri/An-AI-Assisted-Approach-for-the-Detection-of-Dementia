import torch
import os
from config import MAIN_FOLDER

BASE_DIR = MAIN_FOLDER

train_path = os.path.join(BASE_DIR, "precomputed_train.pt")
val_path   = os.path.join(BASE_DIR, "precomputed_val.pt")
out_path   = os.path.join(BASE_DIR, "precomputed_all.pt")

print("Loading train data...")
train = torch.load(train_path)

print("Loading val data...")
val = torch.load(val_path)

print("Merging datasets...")

merged = {
    "spectros": torch.cat([train["spectros"], val["spectros"]], dim=0),
    "input_ids": torch.cat([train["input_ids"], val["input_ids"]], dim=0),
    "attention_mask": torch.cat([train["attention_mask"], val["attention_mask"]], dim=0),
    "labels": torch.cat([train["labels"], val["labels"]], dim=0),
    "file_names": train["file_names"] + val["file_names"],
    "raw_texts": train["raw_texts"] + val["raw_texts"],
}

torch.save(merged, out_path)

print(f"Saved merged dataset → {out_path}")
print(f"Total samples: {len(merged['labels'])}")

data = torch.load("5. Dementia Model/precomputed_all.pt")
print(data["spectros"].shape)
print(data["input_ids"].shape)
print(data["labels"].shape)