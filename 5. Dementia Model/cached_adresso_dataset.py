import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')

CACHE_FILES = {
    "train": "precomputed_train.pt",
    # "val": "precomputed_train.pt",  
    "test": "precomputed_test.pt"
}

class CachedAdressoDataset(Dataset):
    def __init__(self, phase):
        assert phase in CACHE_FILES, f"Invalid phase: {phase}"
        cache_path = CACHE_FILES[phase]
        print(f"🔹 Loading cached dataset: {cache_path}")
        data = torch.load(cache_path)

        self.spectros = data['spectros']
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.labels = data['labels']
        self.file_names = data['file_names']

    def __getitem__(self, idx):
        return {
            'pixels': self.spectros[idx],
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],
            'file_names': self.file_names[idx]
        }

    def __len__(self):
        return len(self.labels)

def variable_batcher(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'pixels': torch.stack([item['pixels'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch])
    }

def adresso_loader(phase, batch_size, shuffle=False):
    dataset = CachedAdressoDataset(phase)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if phase == "train" else False,
        collate_fn=variable_batcher,
        num_workers=4,
        pin_memory=True
    )
