from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from data_loader_vit import adresso_loader 
from bert_image import BertImage 
import argparse

class Trainer:
  def __init__(self, args): 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    os.makedirs('ckp', exist_ok=True) 
    os.makedirs('saved_models', exist_ok=True) 


    torch.manual_seed(args.seed) # Sets the seed for PyTorch's CPU random number generator for reproducibility.
    torch.cuda.manual_seed(args.seed) # Sets the seed for PyTorch's CUDA random number generator for reproducibility.
    np.random.seed(args.seed) # Sets the seed for NumPy's random number generator for reproducibility.

    train_loader = adresso_loader(phase='train', batch_size=args.batch_size, shuffle=True) 
    test_loader = adresso_loader(phase='test', batch_size=args.val_batch_size)

    if torch.cuda.device_count() > 0:
      print(f"{torch.cuda.device_count()} GPUs found") 

    model = BertImage(nfinetune=args.nfinetune) 

    model = nn.DataParallel(model) 
    model.to(device) # Moves the entire model to the specified device.
    params = model.parameters() # Gets an iterator over all learnable parameters of the model.

    optimizer = AdamW(params, lr=args.lr, weight_decay=0.01) # Initializes the AdamW optimizer with the model's parameters, learning rate, and weight decay.
    self.device = device # Stores the selected device as an instance variable.
    self.model = model # Stores the initialized model as an instance variable.
    self.optimizer = optimizer # Stores the optimizer as an instance variable.
    # Changed to CrossEntropyLoss for multi-class classification
    self.multi_class_loss = nn.CrossEntropyLoss()    
    self.train_loader = train_loader # Stores the training data loader.
    self.test_loader = test_loader # Stores the test data loader.
    self.args = args # Stores the arguments object for easy access to configuration.
    self.epoch_accuracies = [] # Initializes an empty list to store accuracies for each epoch.
    self.all_losses = [] # Initializes an empty list to store all batch losses during training.

  def train(self): 
    best_acc = 0.0 # Initializes a variable to track the best accuracy achieved during training.
    best_epoch = 0 # Initializes a variable to track the epoch with the best performance.
    for epoch in range(self.args.epochs): # Loops through the specified number of training epochs.
      print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}") # Prints a separator for the current epoch.
      loss = self.train_epoch() # Calls the train_epoch method to perform one full training pass over the dataset and gets the average epoch loss.
      multi_acc, _, _ = self.eval() 
      print(f"Epoch {epoch + 1} Loss: {loss:.3f}") # Prints the average loss for the current epoch.
      print(f'Multi-Task Acc: {multi_acc:.3f}')
      self.epoch_accuracies.append(round(multi_acc, 3)) 
      
      if multi_acc > best_acc:
            best_acc = multi_acc
            best_epoch = epoch + 1
            torch.save(self.model.state_dict(), f'saved_models/best_model.pth')
            print(f"Best model saved at epoch {best_epoch} with accuracy {best_acc:.3f}")

    with open("epoch_acc.txt", "w") as ofile: 
      for eacc in self.epoch_accuracies:
        ofile.write(str(eacc) + "\n")
    with open("losses.txt", "w") as ofile: 
      for eloss in self.all_losses: 
        ofile.write(str(eloss) + "\n") 

  def train_epoch(self): 
      self.model.train()
      epoch_loss = 0
      for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
          self.optimizer.zero_grad()
          pixels = batch['pixels'].to(self.device)
          input_ids = batch["input_ids"].to(self.device)
          attention_mask = batch["attention_mask"].to(self.device)
          
          # Changed target labels to MMSE scores
          mmse_scores = batch["mimse_scores"].to(self.device)
          
          # The model's forward pass now returns 4 logits
          mmse_logits = self.model(pixels, input_ids, attention_mask)
          
          # Used CrossEntropyLoss for multi-class targets
          loss = self.multi_class_loss(mmse_logits, mmse_scores)
          
          loss.backward()
          self.optimizer.step()
          interval = max(len(self.train_loader) // 20, 1)
          if i % interval == 0 or i == len(self.train_loader) - 1:
              lloss = round(loss.item(), 3)
              print(f'Batch: {i + 1}/{len(self.train_loader)}\ttotal loss: {lloss:.3f}')
              self.all_losses.append(lloss)
          epoch_loss += loss.item()
      return epoch_loss / len(self.train_loader)

  def eval(self):
      self.model.eval()
      mmse_pred = []
      mmse_true = [] 
      loader = self.test_loader

      with torch.no_grad():
          for i, batch in enumerate(loader):
              pixels = batch['pixels'].to(self.device)
              input_ids = batch["input_ids"].to(self.device)
              attention_mask = batch["attention_mask"].to(self.device)
              
              # Getting the MMSE scores as the true labels
              mmse_scores = batch['mimse_scores']
              
              # The model returns 4 logits for each sample
              mmse_logits = self.model(pixels, input_ids, attention_mask)
              
              # Getting the predicted class by finding the index of the max logit
              predicted_classes = torch.argmax(mmse_logits, dim=1)
              
              mmse_pred += predicted_classes.detach().to('cpu').tolist()
              mmse_true += mmse_scores.detach().to('cpu').tolist()
      
      print(mmse_true, "true MMSE bins")
      print(mmse_pred, "pred MMSE bins")
      
      mmse_acc = accuracy_score(mmse_true, mmse_pred)
      
      # Changed target names to reflect the MMSE bins
      target_names = ['Severe (0-9)', 'Moderate (10-18)', 'Mild (19-23)', 'Normal (24-30)']
      print(classification_report(mmse_true, mmse_pred, target_names=target_names, labels=[0, 1, 2, 3], zero_division=0))
      return mmse_acc, mmse_true, mmse_pred
  
  def test(self):
    self.model.load_state_dict(torch.load('saved_models/best_model.pth'))
    final_acc, mmse_true, mmse_pred = self.eval()
    cm = confusion_matrix(mmse_true, mmse_pred)
    print("Confusion Matrix:\n", cm)
    print(f"Final Test Accuracy: {final_acc:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser() # Creates an ArgumentParser object to define command-line arguments.
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of training') # Adds an argument for training batch size.
    parser.add_argument('--val_batch_size', type=int, default=32, help='batch size of testing') # Adds an argument for validation/testing batch size.
    parser.add_argument('--epochs', type=int, default=30) # Adds an argument for the number of training epochs.
    parser.add_argument('--gpu', type=str, default='', help='GPUs to use') # Adds an argument for specifying which GPUs to use.
    parser.add_argument('--lr', type=float, default=3e-5) # Adds an argument for the learning rate.
    parser.add_argument('--nfinetune', type=int, default=48) # Adds an argument for the number of BERT layers to fine-tune.
    parser.add_argument('--seed', type=int, default=0) # Adds an argument for the random seed.
    args = parser.parse_args() # Parses the command-line arguments provided by the user.

    engine = Trainer(args) # Creates an instance of the Trainer class, passing the parsed arguments.
    engine.train() # Starts the training process by calling the train method of the Trainer instance.
    engine.test()