from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from cached_adresso_dataset import CachedAdressoDataset, adresso_loader, variable_batcher 
from bert_image import BertImage 
import argparse
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

class Trainer:
  def __init__(self, args): 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    os.makedirs('ckp', exist_ok=True) 
    os.makedirs('saved_models', exist_ok=True) 

    torch.manual_seed(args.seed) # Sets the seed for PyTorch's CPU random number generator for reproducibility.
    torch.cuda.manual_seed(args.seed) # Sets the seed for PyTorch's CUDA random number generator for reproducibility.
    np.random.seed(args.seed) # Sets the seed for NumPy's random number generator for reproducibility.

    # train_loader = adresso_loader(phase='train', batch_size=args.batch_size) 
    # test_loader = adresso_loader(phase='test', batch_size=args.val_batch_size) 

    if torch.cuda.device_count() > 0:
      print(f"{torch.cuda.device_count()} GPUs found") 

    model = BertImage(nfinetune=args.nfinetune) # Instantiates the BertImage model, passing the number of BERT layers to fine-tune.

    model = nn.DataParallel(model) # Wraps the model with nn.DataParallel. This allows the model to be run on multiple GPUs in parallel if available, distributing the batch across them.
    model.to(device) # Moves the entire model to the specified device.
    params = model.parameters() # Gets an iterator over all learnable parameters of the model.

    optimizer = AdamW(params, lr=args.lr, weight_decay=0.01) # Initializes the AdamW optimizer with the model's parameters, learning rate, and weight decay.
    self.device = device # Stores the selected device as an instance variable.
    self.model = model # Stores the initialized model as an instance variable.
    self.optimizer = optimizer # Stores the optimizer as an instance variable.
    self.binary_cross = nn.BCELoss() # Initializes the Binary Cross-Entropy Loss function, suitable for binary classification problems.
    self.train_loader = None # Stores the training data loader.
    self.test_loader = None # Stores the test data loader.
    self.args = args # Stores the arguments object for easy access to configuration.
    self.epoch_accuracies = [] 
    self.all_losses = [] 

  def train(self): 
    best_acc = 0.0
    best_epoch = 0 # Initializes a variable to track the epoch with the best performance.
    for epoch in range(self.args.epochs): # Loops through the specified number of training epochs.
      print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}") # Prints a separator for the current epoch.
      loss = self.train_epoch() 
      binary_acc, _, _ = self.eval() # Calls the eval method to evaluate the model on the test set and gets the binary classification accuracy.
      print(f"Epoch {epoch + 1} Loss: {loss:.3f}") # Prints the average loss for the current epoch.
      print(f'Binary Acc: {binary_acc:.3f}') # Prints the binary classification accuracy for the current epoch.
      self.epoch_accuracies.append(round(binary_acc, 3)) # Appends the rounded binary accuracy to the list of epoch accuracies.
      
      if binary_acc > best_acc:
            best_acc = binary_acc
            best_epoch = epoch + 1
            torch.save(self.model.state_dict(), f'saved_models/best_model.pth')
            print(f" Saved new best model at epoch {best_epoch} with acc {best_acc:.3f}")
      
    with open("epoch_acc.txt", "w") as ofile:
      for num, eacc in enumerate(self.epoch_accuracies): # Iterates through recorded epoch accuracies.
        ofile.write(f"Epoch {num + 1} accuracy: {eacc}\n") # Writes each accuracy to the file, followed by a newline.
    with open("losses.txt", "w") as ofile:
      for num, eloss in enumerate(self.all_losses): # Iterates through all recorded batch losses.
        ofile.write(f"Epoch {num+1} loss: {eloss} \n") # Writes each loss to the file.

  def train_epoch(self): 
    self.model.train() # Sets the model to training mode. 
    epoch_loss = 0 # Initializes a variable to accumulate loss for the current epoch.
    for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)): # Iterates through batches from the training data loader, with a progress bar.
        self.optimizer.zero_grad() # Clears the gradients of all optimized tensors. This is crucial at the start of each batch to prevent gradient accumulation from previous batches.
        
        pixels = batch['pixels'].to(self.device) # Moves the spectrogram image tensor to the specified device.
        input_ids = batch["input_ids"].to(self.device) # Moves the tokenized input IDs tensor to the device.
        attention_mask = batch["attention_mask"].to(self.device) # Moves the attention mask tensor to the device.
        labels = batch["labels"].to(self.device) # Moves the true labels tensor to the device.
        
        label_preds = self.model(pixels, input_ids, attention_mask) # Performs a forward pass through the model to get predicted probabilities (logits).
        
        labels = labels.reshape(-1) # Reshapes the true labels tensor to a 1D tensor.
        label_preds = label_preds.reshape(-1) # Reshapes the predicted probabilities tensor to a 1D tensor.
        
        loss = self.binary_cross(label_preds, labels.float()) # Calculates the binary cross-entropy loss between predictions and true labels. 
        
        loss.backward() # Performs backpropagation: Computes gradients of the loss with respect to all learnable parameters.
        self.optimizer.step() # Updates the model's parameters using the computed gradients and the optimizer's update rule.
        
        interval = max(len(self.train_loader) // 20, 1) # Defines the interval for printing progress updates.
        if i % interval == 0 or i == len(self.train_loader) - 1: # Checks if it's a logging interval or the last batch.
            lloss = round(loss.item(), 3) # Gets the current batch loss and rounds it.
            print(f'Batch: {i + 1}/{len(self.train_loader)}\ttotal loss: {lloss:.3f}') # Prints the current batch number and its total loss.
            self.all_losses.append(lloss) # Appends the rounded batch loss to the list of all losses.
        epoch_loss += loss.item() # Accumulates the loss for the current epoch.
    return epoch_loss / len(self.train_loader) # Returns the average loss for the epoch.

  def eval(self):
    self.model.eval() # Sets the model to evaluation mode. This disables dropout and ensures batch normalization uses running means/variances.
    label_pred = [] # Initializes an empty list to store predicted labels.
    label_true = [] # Initializes an empty list to store true labels.
    
    loader = self.test_loader # Uses the test data loader for evaluation. 

    with torch.no_grad(): # Disables gradient calculations. This is crucial during evaluation to save memory and computation, as we don't need to update model weights.
      for _, batch in enumerate(loader): # Iterates through batches from the test data loader.
        pixels = batch['pixels'].to(self.device) # Moves spectrograms to device.
        input_ids = batch["input_ids"].to(self.device) # Moves input IDs to device.
        attention_mask = batch["attention_mask"].to(self.device) # Moves attention mask to device.
        labels = batch['labels'] # Retrieves true labels kept on CPU initially.
        
        label_preds = self.model(pixels, input_ids, attention_mask) # Performs a forward pass to get predicted probabilities.
        
        label_pred += label_preds.detach().to('cpu').flatten().round().long().tolist() # Detaches predictions from computation graph, moves to CPU, flattens, rounds to the nearest integer, converts to long, and then to a Python list.
        labels = labels.reshape(-1) # Reshapes true labels to 1D.
        label_true += labels.detach().to('cpu').tolist() # Detaches true labels, moves to CPU, and converts to a Python list.
        
    print(label_true, "true labels") 
    print(label_pred, "pred labels")
    
    label_acc = accuracy_score(label_true, label_pred) # Calculates the overall accuracy score.
    target_names = ['NonDementia', 'Dementia'] # Defines the names for the target classes for the classification report.
    print(classification_report(label_true, label_pred, target_names=target_names))
    return label_acc, label_true, label_pred 
  
  def test(self):
    self.model.load_state_dict(torch.load('saved_models/best_model.pth'))
    final_acc, label_true, label_pred = self.eval()
    print(f"Final Test Accuracy: {final_acc:.3f}")
    cm = confusion_matrix(label_true, label_pred)
    print("Confusion Matrix:\n", cm)
    
def run_cv(args):
  dataset = CachedAdressoDataset(phase="all")
  labels = dataset.labels.cpu().numpy()

  skf = StratifiedKFold(
      n_splits=args.n_folds,
      shuffle=True,
      random_state=args.seed
  )

  for fold, (train_idx, val_idx) in enumerate(skf.split(labels, labels)):
      print(f"\n========== Fold {fold+1}/{args.n_folds} ==========")

      train_set = Subset(dataset, train_idx)
      val_set   = Subset(dataset, val_idx)

      train_loader = DataLoader(
          train_set,
          batch_size=args.batch_size,
          shuffle=True,
          collate_fn=variable_batcher,
          num_workers=4,
          pin_memory=True
      )

      val_loader = DataLoader(
          val_set,
          batch_size=args.val_batch_size,
          shuffle=False,
          collate_fn=variable_batcher,
          num_workers=4,
          pin_memory=True
      )

      engine = Trainer(args)
      engine.train_loader = train_loader
      engine.test_loader  = val_loader

      engine.train()

      torch.save(
          engine.model.state_dict(),
          f"saved_models/fold_{fold+1}.pth"
      )

if __name__ == '__main__':
    parser = argparse.ArgumentParser() # Creates an ArgumentParser object to define command-line arguments.
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of training') # Adds an argument for training batch size.
    parser.add_argument('--val_batch_size', type=int, default=32, help='batch size of testing') # Adds an argument for validation/testing batch size.
    parser.add_argument('--epochs', type=int, default=30) # Adds an argument for the number of training epochs.
    parser.add_argument('--gpu', type=str, default='', help='GPUs to use') # Adds an argument for specifying which GPUs to use (e.g., '0,1').
    parser.add_argument('--lr', type=float, default=3e-5) # Adds an argument for the learning rate.
    parser.add_argument('--nfinetune', type=int, default=48) # Adds an argument for the number of BERT layers to fine-tune.
    parser.add_argument('--seed', type=int, default=0) # Adds an argument for the random seed.
    parser.add_argument('--n_folds', type=int, default=5) # Adds an argument for the number of folds for cross-validation.
    args = parser.parse_args() # Parses the command-line arguments provided by the user.

    run_cv(args) # Calls the run_cv function to perform cross-validation training and evaluation.