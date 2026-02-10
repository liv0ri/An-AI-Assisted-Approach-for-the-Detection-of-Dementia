import torch
import os
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
import csv

def preprocess_multimodal_split(
    spectro_dir,
    trans_dir,
    output_path,
    tokenizer_name='FacebookAI/roberta-base',
    label_map=None,
    labels_csv=None,
    max_length=256
):
    """
    Precompute spectrogram tensors + tokenized transcripts for train/val/test.
    Supports:
      - Folder-based labels (ad/cn)
      - CSV-based labels (filename -> label)

    Saves a .pt file with tensors.
    """
    output_dir = "5. Dementia Model"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    spectro_tensors = []
    texts = []
    labels_list = []
    file_names = []

    if labels_csv is None:
        # Folder-based labels
        for label_str, label_id in label_map.items():
            # get the folder path for this label
            folder_path = os.path.join(spectro_dir, label_str)

            # iterate through the files in the folder
            # tqdm - the progress bar https://www.geeksforgeeks.org/python/python-how-to-make-a-terminal-progress-bar-using-tqdm/
            for fname in tqdm(os.listdir(folder_path)):
                if not fname.endswith('.png'):
                    continue

                # get the spectrogram and transcript paths
                spectro_path = os.path.join(folder_path, fname)
                trans_path = os.path.join(trans_dir, label_str, fname.replace('.png', '.txt'))
                # Image
                img = Image.open(spectro_path).convert('RGB')
                # Apply transforms and store tensor
                spectro_tensors.append(img_transform(img))

                # Read text
                with open(trans_path, 'r', encoding='utf-8') as f:
                    # Remove newlines and extra spaces
                    texts.append(f.read().strip().replace('\n', ' '))

                # Label + name
                labels_list.append(label_id)
                file_names.append(fname)
    else:
        # CSV-based labels
        labels_dict = {}
        # Read the CSV file
        with open(labels_csv, newline='') as csvfile:
            # Read the CSV file
            reader = csv.reader(csvfile)
            # skip header - the column names
            next(reader)  
            # iterate through the rows
            for row in reader:
                labels_dict[row[0]] = label_map[row[1]]

        # iterate through the files in the directory
        for fname in tqdm(sorted(os.listdir(spectro_dir))):
            if not fname.endswith('.png'):
                continue

            # get the sample id
            sample_id = fname.replace('.png', '')
            spectro_path = os.path.join(spectro_dir, fname)
            trans_path = os.path.join(trans_dir, sample_id + '.txt')

            # Image
            img = Image.open(spectro_path).convert('RGB')
            # Apply transforms and store tensor
            spectro_tensors.append(img_transform(img))

            # Read text
            with open(trans_path, 'r', encoding='utf-8') as f:
                texts.append(f.read().strip().replace('\n', ' '))

            # Label + name
            labels_list.append(labels_dict[sample_id])
            file_names.append(fname)

    # Tokenize all texts at once
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Stack tenspr
    spectro_tensors = torch.stack(spectro_tensors)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    torch.save({
        'spectros': spectro_tensors,
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels_tensor,
        'file_names': file_names,
    }, output_path)

    print(f"Saved precomputed data to → {output_path}")

preprocess_multimodal_split(
    spectro_dir='diagnosis/train/specto/',
    trans_dir='diagnosis/train/trans/',
    output_path='precomputed_train.pt',
    label_map={"ad": 1, "cn": 0}
)

preprocess_multimodal_split(
    spectro_dir='diagnosis/val/specto/',
    trans_dir='diagnosis/val/trans/',
    output_path='precomputed_val.pt',
    label_map={"ad": 1, "cn": 0}
)

preprocess_multimodal_split(
    spectro_dir='diagnosis/test-distspecto/',
    trans_dir='diagnosis/test-disttrans/',
    output_path='precomputed_test.pt',
    labels_csv='diagnosis/test-dist/task1.csv',
    label_map={"Control": 0, "ProbableAD": 1}
)
