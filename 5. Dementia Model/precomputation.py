import torch
import os
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
import csv

# https://chatgpt.com/share/6903659c-c064-8006-9cbb-0e43a3dc39b3
spectro_dir = 'diagnosis/train/specto/'
trans_dir = 'diagnosis/train/trans/'
label_map = {"ad": 1, "cn": 0} 

# Tokenizer & image transforms
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')
img_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Prepare holders
spectro_tensors = []
# store all transcripts here
texts = []         
labels_list = []
file_names = []

for label_str in ["ad", "cn"]:
    folder_path = os.path.join(spectro_dir, label_str)
    for fname in tqdm(os.listdir(folder_path)):
        if not fname.endswith('.png'):
            continue
        spectro_path = os.path.join(spectro_dir, label_str, fname)
        trans_path = os.path.join(trans_dir, label_str, fname.replace('.png', '.txt'))
        
        # Image
        img = Image.open(spectro_path).convert('RGB')
        spectro_tensor = img_transform(img)
        spectro_tensors.append(spectro_tensor)

        # Read text
        with open(trans_path, 'r', encoding='utf-8') as f:
            text = f.read().strip().replace('\n', ' ')
            texts.append(text)

        # Label + name
        labels_list.append(label_map[label_str])
        file_names.append(fname)

# Tokenize all texts at once
encodings = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

# Stack tensors
spectro_tensors = torch.stack(spectro_tensors)
labels_tensor = torch.tensor(labels_list, dtype=torch.long)

# Save to file
torch.save({
    'spectros': spectro_tensors,
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': labels_tensor,
    'file_names': file_names,
}, 'precomputed_train.pt')


spectro_dir = 'diagnosis/test-distspecto/'
trans_dir = 'diagnosis/test-disttrans/'
labels_csv = 'diagnosis/test-dist/task1.csv'

label_map = {"Control": 0, "ProbableAD": 1}
labels_dict = {}
with open(labels_csv, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip header
    for row in reader:
        # row[0] : filename without extension, row[1] : label string
        labels_dict[row[0]] = label_map[row[1]]

spectro_tensors = []
texts = []          # reset for test
labels_list = []
file_names = []

for fname in tqdm(sorted(os.listdir(spectro_dir))):
    if not fname.endswith('.png'):
        continue

    sample_id = fname.replace('.png', '')
    spectro_path = os.path.join(spectro_dir, fname)
    trans_path = os.path.join(trans_dir, sample_id + '.txt')

    # Image
    img = Image.open(spectro_path).convert('RGB')
    spectro_tensor = img_transform(img)
    spectro_tensors.append(spectro_tensor)
    
    # Transcript (tokenize)
    with open(trans_path, 'r', encoding='utf-8') as f:
        text = f.read().strip().replace('\n', ' ')
        texts.append(text)

    # Label + name
    labels_list.append(labels_dict[sample_id])
    file_names.append(fname)

encodings = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")

# Stack tensors
spectro_tensors = torch.stack(spectro_tensors)
labels_tensor = torch.tensor(labels_list, dtype=torch.long)

# Save to file
torch.save({
    'spectros': spectro_tensors,
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': labels_tensor,
    'file_names': file_names,
}, 'precomputed_test.pt')