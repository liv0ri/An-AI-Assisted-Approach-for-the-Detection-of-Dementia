import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import csv, os
from sklearn.utils import shuffle
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')

transform_list = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def parse_mimse(scores_file):
    mdict = {}
    with open(scores_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            mdict[row[1]] = int(row[2])
    return mdict

def parse_mimse_test(scores_file):
    mdict = {}
    with open(scores_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            mdict[row[0]] = int(row[1])
    return mdict

def parse_labels_test(scores_file):
    mdict = {}
    ldict = {"Control": 0, "ProbableAD":1}
    with open(scores_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            mdict[row[0]] = ldict[row[1]]
    return mdict

mimse_score_bins = [
    ((24, 30), 3),
    ((19, 23), 2),
    ((10, 18), 1),
    ((0, 9), 0)
]

def bin_mimse_score(score):
    for mbin, label in mimse_score_bins:
        (start, end) = mbin
        if start <= score <= end:
            return label
    return 0

base_dir = "../diagnosis/"
class AdressoDataset(Dataset):
    def __init__(self, phase):
        if phase == "train":
            train_dir = os.path.join(base_dir, "train")
            data_path_ad, data_path_cn = os.path.join(train_dir, "audio/ad"), os.path.join(train_dir, "audio/cn")
            specto_path_ad, specto_path_cn = os.path.join(train_dir, "specto/ad"), os.path.join(train_dir, "specto/cn")
            trans_path_ad, trans_path_cn = os.path.join(train_dir, "trans/ad"), os.path.join(train_dir, "trans/cn")
            mimse_file = base_dir + "train/adresso-train-mmse-scores.csv"
            files_ad = [fname for fname in os.listdir(data_path_ad) if fname.endswith(".wav")]
            files_cn = [fname for fname in os.listdir(data_path_cn) if fname.endswith(".wav")]
            specto_files_ad = [os.path.join(specto_path_ad, fname[:-4] + ".png") for fname in files_ad]
            specto_files_cn = [os.path.join(specto_path_cn, fname[:-4] + ".png") for fname in files_cn]
            trans_files_ad = [os.path.join(trans_path_ad, fname[:-4] + ".txt") for fname in files_ad]
            trans_files_cn = [os.path.join(trans_path_cn, fname[:-4] + ".txt") for fname in files_cn]
            all_filen = files_ad + files_cn
            labels = [1] * len(files_ad) + [0] * len(files_cn)
            mimse_dict = parse_mimse(mimse_file)
            mimse_scores = [mimse_dict[filen[:-4]] for filen in all_filen]
            mimse_scores = [bin_mimse_score(score) for score in mimse_scores]
            specto_filen = specto_files_ad + specto_files_cn
            trans_filen = trans_files_ad + trans_files_cn
            self.all_filen, self.specto_filen, self.trans_filen, self.labels, self.mimse_scores = shuffle(all_filen, specto_filen, trans_filen, labels, mimse_scores, random_state=44)
            self.mimse_scores = torch.tensor(self.mimse_scores, dtype=torch.long)
        elif phase == "test":
            test_dir = os.path.join(base_dir, "test-dist")
            data_path = os.path.join(test_dir + "audio")
            specto_path = os.path.join(test_dir + "specto/")
            trans_path = os.path.join(test_dir + "trans/")
            labels_file = os.path.join(test_dir, "task1.csv" )
            mimse_file = os.path.join(test_dir, "task2.csv")
            all_filen = [fname for fname in os.listdir(data_path) if fname.endswith(".wav")]
            specto_files = [os.path.join(specto_path, fname[:-4] + ".png") for fname in all_filen]
            trans_files = [os.path.join(trans_path, fname[:-4] + ".txt") for fname in all_filen]
            mimse_dict = parse_mimse_test(mimse_file)
            mimse_scores = [mimse_dict[filen[:-4]] for filen in all_filen]
            mimse_scores = [bin_mimse_score(score) for score in mimse_scores]
            labels_dict = parse_labels_test(labels_file)
            labels = [labels_dict[filen[:-4]] for filen in all_filen]
            self.all_filen, self.specto_filen, self.trans_filen, self.labels, self.mimse_scores = all_filen, specto_files, trans_files, labels, mimse_scores
            self.mimse_scores = torch.tensor(self.mimse_scores, dtype=torch.long)

    def __getitem__(self, index):
        specto_path = self.specto_filen[index]
        specto = Image.open(specto_path).convert('RGB')
        pixels = transform_list(specto)
        pixels = pixels.unsqueeze(0)
        transcript_path = self.trans_filen[index]
        transcript = open(transcript_path, "r", encoding="utf-8").read()
        transcript = " ".join(transcript.strip().split())
        if "train" in specto_path:
            specto = transform_list(specto)
        item = {
            'spectos': specto,
            'pixels': pixels,
            "file_names": self.all_filen[index],
            'mimse_scores': self.mimse_scores[index],
            'labels': self.labels[index],
            'transcripts': transcript
        }
        return item

    def __len__(self):
        return len(self.labels)

def variable_batcher(batch):
    all_texts = [item["transcripts"] for item in batch]
    encodings = tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels = [item["labels"] for item in batch]
    mmse_scores = [item["mimse_scores"] for item in batch]
    pixels_tc = [item["pixels"] for item in batch]
    pixels = torch.stack(pixels_tc)
    pixels = pixels.squeeze(1)
    item = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixels': pixels,
        'labels': torch.tensor(labels, dtype=torch.long),
        'mmse_scores': torch.tensor(mmse_scores, dtype=torch.long)
    }
    return item

def adresso_loader(phase, batch_size, shuffle=False):
    dataset = AdressoDataset(phase)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=variable_batcher)