import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import csv, os
from sklearn.utils import shuffle
from config import output_dir, test_csv_name
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')

transform_list = transforms.Compose([ 
    transforms.Resize(256), # Resizes the input image spectrogram so that its shorter side is 256 pixels, maintaining the aspect ratio.
    transforms.CenterCrop(224), # Crops the center of the image to a square of 224x224 pixels.
    transforms.ToTensor(), # Converts the PIL Image or NumPy ndarray into a PyTorch FloatTensor and scales pixel intensities to the range.
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizes the tensor image with a given mean and standard deviation for each color channel. 
])

def parse_labels_test(scores_file): 
  mdict = {} 
  ldict = {"Control": 0, "ProbableAD":1} 
  with open(scores_file, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',') 
    next(reader) 
    for row in reader: 
      mdict[row[0]] = ldict[row[1]] # Assigns the numerical label based on the string label in the second column to the key from the first column.
  return mdict 

base_dir = f"../{output_dir}/" 
class AdressoDataset(Dataset): 
    def __init__(self, phase): 
      if phase == "train": 
        train_dir = os.path.join(base_dir, "train")
        data_path_ad, data_path_cn = os.path.join(train_dir, "audio/ad"), os.path.join(train_dir, "audio/cn") # Constructs paths to audio for AD and CN groups.

        specto_path_ad, specto_path_cn = os.path.join(train_dir, "specto/ad"), os.path.join(train_dir, "specto/cn") # Constructs paths to spectrogram image files for AD and CN groups.

        trans_path_ad, trans_path_cn = os.path.join(train_dir, "trans/ad"), os.path.join(train_dir, "trans/cn") # Constructs paths to transcription text files for AD and CN groups.

        files_ad = [fname for fname in os.listdir(data_path_ad) if fname.endswith(".mp3")] # Lists all files in the AD audio directory that end with '.png'.
        files_cn = [fname for fname in os.listdir(data_path_cn) if fname.endswith(".mp3")] # Lists all files in the CN audio directory that end with '.png'.

        specto_files_ad = [os.path.join(specto_path_ad, fname[:-4] + ".png") for fname in files_ad] # Creates full paths for AD spectrogram files.
        specto_files_cn = [os.path.join(specto_path_cn, fname[:-4] + ".png") for fname in files_cn] # Creates full paths for CN spectrogram files.        
        trans_files_ad = [os.path.join(trans_path_ad, fname[:-4] + ".txt") for fname in files_ad] # Creates full paths for AD transcription files by changing '.png' to '.txt' in the filename.
        trans_files_cn = [os.path.join(trans_path_cn, fname[:-4] + ".txt") for fname in files_cn] # Creates full paths for CN transcription files.

        all_filen = files_ad + files_cn # Combines the lists of AD and CN filenames.
        labels = [1] * len(files_ad) + [0] * len(files_cn) # Creates a list of numerical labels: 1 for AD samples, 0 for Control samples.
        specto_filen = specto_files_ad + specto_files_cn # Combines the lists of AD and CN spectrogram file paths.
        trans_filen =  trans_files_ad + trans_files_cn # Combines the lists of AD and CN transcription file paths.

        self.all_filen, self.specto_filen, self.trans_filen, self.labels = shuffle(all_filen, specto_filen, trans_filen, labels, random_state=44)

      elif phase  == "test":
        test_dir = os.path.join(base_dir, "test-dist") # Constructs the full path to the distributed test data directory.
        data_path = os.path.join(test_dir + "audio") 
        specto_path = os.path.join(test_dir + "specto/") # Constructs path to spectrogram image files in the test set.
        trans_path = os.path.join(test_dir + "trans/") # Constructs path to transcription text files in the test set.

        labels_file =  os.path.join(test_dir, f"{test_csv_name}" ) # Defines the full path to the test diagnostic labels CSV file.

        all_filen = [fname for fname in os.listdir(data_path) if fname.endswith(".mp3")] # Lists all '.png' files in the test audio directory.
        specto_files = [os.path.join(specto_path, fname[:-4] + ".png") for fname in all_filen] 
        trans_files = [os.path.join(trans_path, fname[:-4] + ".txt") for fname in all_filen]

        labels_dict = parse_labels_test(labels_file) # Parses the test diagnostic labels CSV file.
        labels = [labels_dict[filen[:-4]] for filen in all_filen] # Retrieves diagnostic labels for test filenames.

        self.all_filen, self.specto_filen, self.trans_filen, self.labels = all_filen, specto_files, trans_files, labels # Assigns the prepared lists to instance variables. No shuffling for the test set.

    def __getitem__(self, index): # This method is called by the DataLoader to retrieve a single sample at a given index.
      specto_path = self.specto_filen[index] # Gets the spectrogram file path for the current index.
      specto = Image.open(specto_path).convert('RGB') # Opens the spectrogram image using PIL and converts it to RGB format.
      pixels = transform_list(specto) # Applies the defined transformation list to the spectrogram image.
      pixels = pixels.unsqueeze(0) # Adds an extra dimension at the beginning of the `pixels` tensor. 

      transcript_path = self.trans_filen[index] # Gets the transcription file path for the current index.
      transcript = open(transcript_path, "r", encoding="utf-8").read() # Opens and reads the entire content of the transcription text file.
      transcript = " ".join(transcript.strip().split())

      item = { 
        'spectos': specto, # Stores the transformed spectrogram tensor.
        'pixels': pixels, # Stores the transformed spectrogram tensor with an added dimension. 
        "file_names": self.all_filen[index], # Stores the original filename.
        'labels': self.labels[index], # Stores the diagnostic label for this sample.
        'transcripts': transcript # Stores the cleaned text transcript.
      }
      return item

    def __len__(self):
      return len(self.labels) 

def variable_batcher(batch): 
  all_texts = [item["transcripts"] for item in batch] # Extracts all raw transcripts from the list of individual sample dictionaries.
  encodings = tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt")  
  input_ids = encodings['input_ids'] # Extracts the tokenized input IDs (numerical representations of words) from the tokenizer's output.
  attention_mask = encodings['attention_mask'] # Extracts the attention mask, which indicates which tokens are real and which are padding.

  labels = [item["labels"] for item in batch] # Extracts all diagnostic labels from the batch.

  pixels_tc = [item["pixels"] for item in batch] # Extracts all spectrogram tensors from the batch.
  pixels = torch.stack(pixels_tc) # Stacks the list of individual spectrogram tensors into a single batch tensor. This adds the batch dimension.
  pixels = pixels.squeeze(1) # Removes the redundant dimension from the stacked pixels tensor. The final shape will be (batch_size, C, H, W).
  item = { 
      'input_ids': input_ids, # The batched tensor of tokenized input IDs.
      'attention_mask': attention_mask, # The batched tensor of attention masks.
      'pixels': pixels, # The batched tensor of processed spectrograms.
      'labels': torch.tensor(labels,  dtype=torch.long) # Converts the list of labels into a PyTorch tensor of long integers.
    }
  return item

def adresso_loader(phase, batch_size, shuffle=False):
  dataset = AdressoDataset(phase)
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=variable_batcher) 