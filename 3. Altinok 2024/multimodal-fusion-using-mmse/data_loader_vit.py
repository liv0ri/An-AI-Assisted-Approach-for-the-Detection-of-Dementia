import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import csv, os
from sklearn.utils import shuffle
tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-base')

transform_list = transforms.Compose([ 
    transforms.Resize(256), # Resizes the input image spectrogram so that its shorter side is 256 pixels, maintaining the aspect ratio.
    transforms.CenterCrop(224), # Crops the center of the image to a square of 224x224 pixels.
    transforms.ToTensor(), # Converts the PIL Image or NumPy ndarray into a PyTorch FloatTensor and scales pixel intensities to the [0.0, 1.0] range.
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizes the tensor image with a given mean and standard deviation for each color channel. 
])

def parse_mimse(scores_file): 
  mdict = {} 
  with open(scores_file, newline='') as csvfile: 
    reader = csv.reader(csvfile, delimiter=',')
    next(reader) # Skips the header row of the CSV file.
    for row in reader: # Iterates over each subsequent row in the CSV file.
      mdict[row[1]] = int(row[2])
  return mdict 
   
def parse_mimse_test(scores_file): 
  mdict = {}
  with open(scores_file, newline='') as csvfile: 
    reader = csv.reader(csvfile, delimiter=',') 
    next(reader) # Skips the header row.
    for row in reader: # Iterates through each row.
      mdict[row[0]] = int(row[1]) # Assigns the integer value from the second column as the MMSE score to the key from the first column for the test set format.
  return mdict 

def parse_labels_test(scores_file): 
  mdict = {} 
  ldict = {"Control": 0, "ProbableAD":1} # Defines a mapping from string labels to numerical labels.
  with open(scores_file, newline='') as csvfile: # Opens the test labels CSV file.
    reader = csv.reader(csvfile, delimiter=',') 
    next(reader) # Skips the header row.
    for row in reader: # Iterates through each row.
      mdict[row[0]] = ldict[row[1]] # Assigns the numerical label based on the string label in the second column to the key from the first column.
  return mdict 

mimse_score_bins = [ 
    ((24, 30), 3), # Represents scores 24-30 as "normal cognition" (label 3).
    ((19, 23), 2), # Represents scores 19-23 as "mild cognitive impairment" (label 2).
    ((10, 18), 1), # Represents scores 10-18 as "moderate impairment" (label 1).
    ((0, 9), 0) # Represents scores 0-9 as "severe impairment" (label 0).
]

def bin_mimse_score(score): 
  for mbin, label in mimse_score_bins: # Iterates through each defined MMSE score bin.
    (start, end) = mbin # Unpacks the start and end values of the current score range.
    if start <= score <= end: # Checks if the given score falls within the current bin's range.
      return label # If it does, returns the corresponding categorical label for that bin.
  return 0

base_dir = "../diagnosis/" 
class AdressoDataset(Dataset): 
    def __init__(self, phase): 
      if phase == "train": 
        train_dir = os.path.join(base_dir, "train") # Constructs the full path to the training data directory.
        data_path_ad, data_path_cn = os.path.join(train_dir, "audio/ad"), os.path.join(train_dir, "audio/cn") # Constructs paths to audio for Alzheimer's Disease (AD) and Control (CN) groups.

        specto_path_ad, specto_path_cn = os.path.join(train_dir, "specto/ad"), os.path.join(train_dir, "specto/cn") # Constructs paths to spectrogram image files for AD and CN groups.

        trans_path_ad, trans_path_cn = os.path.join(train_dir, "trans/ad"), os.path.join(train_dir, "trans/cn") # Constructs paths to transcription text files for AD and CN groups.

        mimse_file =  base_dir + "train/adresso-train-mmse-scores.csv" # Defines the full path to the MMSE scores CSV file for the training set.

        files_ad = [fname for fname in os.listdir(data_path_ad) if fname.endswith(".wav")] 
        files_cn = [fname for fname in os.listdir(data_path_cn) if fname.endswith(".wav")] 

        specto_files_ad = [os.path.join(specto_path_ad, fname[:-4] + ".png") for fname in files_ad]
        specto_files_cn = [os.path.join(specto_path_cn, fname[:-4] + ".png") for fname in files_cn]
        trans_files_ad = [os.path.join(trans_path_ad, fname[:-4] + ".txt") for fname in files_ad] 
        trans_files_cn = [os.path.join(trans_path_cn, fname[:-4] + ".txt") for fname in files_cn] 

        all_filen = files_ad + files_cn # Combines the lists of AD and CN filenames.
        labels = [1] * len(files_ad) + [0] * len(files_cn) # Creates a list of numerical labels: 1 for AD samples, 0 for Control samples.
        mimse_dict = parse_mimse(mimse_file) # Parses the training MMSE scores CSV file into a dictionary.
        mimse_scores = [mimse_dict[filen[:-4]] for filen in all_filen] # Retrieves raw MMSE scores for all filenames based on the parsed dictionary. The [:-4] removes the '.png' extension.
        mimse_scores = [bin_mimse_score(score) for score in mimse_scores] # Converts raw MMSE scores into binned categorical labels.
        specto_filen = specto_files_ad + specto_files_cn # Combines the lists of AD and CN spectrogram file paths.
        trans_filen =  trans_files_ad + trans_files_cn # Combines the lists of AD and CN transcription file paths.

        self.all_filen, self.specto_filen, self.trans_filen, self.labels, self.mimse_scores = shuffle(all_filen, specto_filen, trans_filen, labels, mimse_scores, random_state=44) # Shuffles all corresponding lists consistently using a fixed random_state for reproducibility.
        self.mimse_scores = torch.tensor(self.mimse_scores, dtype=torch.long) # Converts the list of binned MMSE scores into a PyTorch tensor of long integers.

      elif phase  == "test": # Checks if the dataset is being initialized for the test phase.
        test_dir = os.path.join(base_dir, "test-dist") # Constructs the full path to the distributed test data directory.
        data_path = os.path.join(test_dir + "audio") # Constructs path to audio files in the test set.
        specto_path = os.path.join(test_dir + "specto/") # Constructs path to spectrogram image files in the test set.
        trans_path = os.path.join(test_dir + "trans/") # Constructs path to transcription text files in the test set.

        labels_file =  os.path.join(test_dir, "task1.csv" ) # Defines the full path to the test diagnostic labels CSV file.
        mimse_file =  os.path.join(test_dir,  "task2.csv") # Defines the full path to the test MMSE scores CSV file.

        all_filen = [fname for fname in os.listdir(data_path) if fname.endswith(".wav")] # Lists all '.png' files in the test audio directory.
        specto_files = [os.path.join(specto_path, fname[:-4] + ".png") for fname in all_filen]
        trans_files = [os.path.join(trans_path, fname[:-4] + ".txt") for fname in all_filen] # Creates full paths for test transcription files, replacing '.png' with '.txt'.

        mimse_dict = parse_mimse_test(mimse_file) 
        mimse_scores = [mimse_dict[filen[:-4]] for filen in all_filen] # Retrieves raw MMSE scores for test filenames.
        mimse_scores = [bin_mimse_score(score) for score in mimse_scores] # Converts test MMSE scores into binned categorical labels.
        labels_dict = parse_labels_test(labels_file) # Parses the test diagnostic labels CSV file.
        labels = [labels_dict[filen[:-4]] for filen in all_filen] # Retrieves diagnostic labels for test filenames.

        self.all_filen, self.specto_filen, self.trans_filen, self.labels, self.mimse_scores = all_filen, specto_files, trans_files, labels, mimse_scores # Assigns the prepared lists to instance variables. 
        self.mimse_scores = torch.tensor(self.mimse_scores, dtype=torch.long) # Converts test binned MMSE scores to a PyTorch tensor.

    def __getitem__(self, index): # This method is called by the DataLoader to retrieve a single sample at a given index.
      specto_path = self.specto_filen[index] # Gets the spectrogram file path for the current index.
      specto = Image.open(specto_path).convert('RGB') 
      pixels = transform_list(specto) # Applies the defined transformation list (resize, crop, to tensor, normalize) to the spectrogram image.
      pixels = pixels.unsqueeze(0) # Adds an extra dimension at the beginning of the `pixels` tensor. This typically makes its shape (1, C, H, W), preparing it for potential batching.

      transcript_path = self.trans_filen[index] # Gets the transcription file path for the current index.
      transcript = open(transcript_path, "r", encoding="utf-8").read() # Opens and reads the entire content of the transcription text file.
      transcript = " ".join(transcript.strip().split()) # Cleans the transcript: .strip() removes leading/trailing whitespace, .split() splits it into words, and " ".join() rejoins them with single spaces, collapsing multiple spaces.
      
      if "train" in specto_path: 
        specto =  transform_list(specto) # If "train" is in the path, it re-applies the transformations to `specto`. This line seems redundant as `pixels` already holds the transformed version of `specto`.

      item = { 
        'spectos': specto, # Stores the transformed spectrogram tensor (from the potentially redundant line above, or the original transformed `specto` before `unsqueeze`).
        'pixels': pixels, # Stores the transformed spectrogram tensor with an added dimension. This is likely the intended image input.
        "file_names": self.all_filen[index], # Stores the original filename.
        'mimse_scores': self.mimse_scores[index], # Stores the binned MMSE score for this sample.
        'labels': self.labels[index], # Stores the diagnostic label for this sample.
        'transcripts': transcript # Stores the cleaned text transcript.
      }
      return item

    def __len__(self):
      return len(self.labels) 

def variable_batcher(batch): 
  all_texts = [item["transcripts"] for item in batch] # Extracts all raw transcripts from the list of individual sample dictionaries.
  encodings = tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt") # Tokenizes all transcripts in the batch: padding=True pads shorter sequences to the length of the longest in the batch.
  input_ids = encodings['input_ids'] # Extracts the tokenized input IDs (numerical representations of words) from the tokenizer's output.
  attention_mask = encodings['attention_mask'] # Extracts the attention mask, which indicates which tokens are real and which are padding.
  mmse_scores = [item["mimse_scores"] for item in batch]
  
  pixels_tc = [item["pixels"] for item in batch]
  pixels = torch.stack(pixels_tc)
  pixels = pixels.squeeze(1)

  item = {
      'input_ids': input_ids,
      'attention_mask': attention_mask,
      'pixels': pixels,
      'mimse_scores': torch.tensor(mmse_scores, dtype=torch.long) # Add the MMSE scores to the batch
  }
  return item

def adresso_loader(phase, batch_size, shuffle=False):
  dataset = AdressoDataset(phase) 
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=variable_batcher) 