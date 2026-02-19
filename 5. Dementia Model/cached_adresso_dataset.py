import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')

CACHE_FILES = {
    "train": "precomputed_train.pt",
    "val": "precomputed_val.pt",  
    "test": "precomputed_test.pt"
}

class CachedAdressoDataset(Dataset):
    def __init__(self, phase, base_path=""):
        assert phase in CACHE_FILES, f"Invalid phase: {phase}"
        cache_path = CACHE_FILES[phase]
        print(f"Loading cached dataset: {cache_path}")
        cache_full_path = os.path.join(base_path, cache_path)
        data = torch.load(cache_full_path)

        self.spectros = data['spectros']
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.labels = data['labels']
        self.file_names = data['file_names']
        self.raw_texts = data['raw_texts'] 

    def __getitem__(self, idx):
        return {
            'pixels': self.spectros[idx],
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
            'file_names': self.file_names[idx],
            'raw_texts': self.raw_texts[idx]
        }

    def __len__(self):
        return len(self.labels)

def variable_batcher(batch):
    out = {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'pixels': torch.stack([item['pixels'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }
    if 'raw_texts' in batch[0]:
        out['raw_texts'] = [item['raw_texts'] for item in batch]
    if 'file_names' in batch[0]:
        out['file_names'] = [item['file_names'] for item in batch]
    return out

def adresso_loader(phase, batch_size, base_path=""):
    dataset = CachedAdressoDataset(phase, base_path=base_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if phase == "train" else False,
        collate_fn=variable_batcher,
        num_workers=4,
        pin_memory=True
    )
