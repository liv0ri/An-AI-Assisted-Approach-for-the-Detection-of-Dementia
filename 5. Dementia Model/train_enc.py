import csv
import pandas as pd
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from cached_adresso_dataset import CachedAdressoDataset, variable_batcher 
from bert_image import BertImage 
import argparse
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset
from captum.attr import IntegratedGradients, LayerIntegratedGradients
import matplotlib.pyplot as plt

class Trainer:
  def __init__(self, args, fold): 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    os.makedirs('ckp', exist_ok=True) 
    os.makedirs('saved_models', exist_ok=True) 

    torch.manual_seed(args.seed) # Sets the seed for PyTorch's CPU random number generator for reproducibility.
    torch.cuda.manual_seed(args.seed) # Sets the seed for PyTorch's CUDA random number generator for reproducibility.
    np.random.seed(args.seed) # Sets the seed for NumPy's random number generator for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    self.val_loader = None # Stores the validation data loader.
    self.args = args # Stores the arguments object for easy access to configuration.
    self.epoch_accuracies = [] 
    self.all_losses = [] 
    self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    self.fold = fold
    self.best_threshold = 0.5

    model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    # Integrated Gradients
    self.ig_audio = IntegratedGradients(model_ref)

    self.lig_text = LayerIntegratedGradients(
        self.text_forward,
        model_ref.bert.embeddings
    )
    os.makedirs("xai", exist_ok=True)  # folder to save IG images

  def train(self): 
    best_f1 = 0.0
    best_epoch = 0 # Initializes a variable to track the epoch with the best performance.
    for epoch in range(self.args.epochs): # Loops through the specified number of training epochs.
      print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}") # Prints a separator for the current epoch.
      loss = self.train_epoch() 
      _, _, label_f1, _ = self.eval() # Calls the eval method to evaluate the model on the val set and gets the auc.
      print(f'Macro F1 Score: {label_f1:.3f}')

      print(f"Epoch {epoch + 1} Loss: {loss:.3f}") # Prints the average loss for the current epoch.
      print(f'Macro F1 Score: {label_f1:.3f}') 
      self.epoch_accuracies.append(round(label_f1, 3)) # Appends the rounded binary accuracy to the list of epoch accuracies.
      
      if label_f1 > best_f1:
            self.xai_step(epoch=epoch, num_samples=3)
            best_f1 = label_f1
            best_epoch = epoch + 1
            torch.save(
                self.model.state_dict(),
                f'saved_models/best_model_fold{self.fold}.pth'
            )
            print(f" Saved new best model at epoch {best_epoch} with f1 {best_f1:.3f}")
      
    with open(f"epoch_acc_fold{self.fold}.txt", "w") as ofile:
      for num, eacc in enumerate(self.epoch_accuracies): # Iterates through recorded epoch accuracies.
        ofile.write(f"Epoch {num + 1} accuracy: {eacc}\n") # Writes each accuracy to the file, followed by a newline.
    with open(f"losses_fold{self.fold}.txt", "w") as ofile:
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
  
  def text_forward(self, input_ids, pixels, attention_mask):
    return self.model(pixels, input_ids, attention_mask)
  
  def xai_step(self, epoch, num_samples=5):
    self.model.eval()
    count = 0

    for batch in self.val_loader:
        pixels = batch['pixels'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels']

        for i in range(len(labels)):
            if count >= num_samples:
                return  

            pixel_input = pixels[i:i+1]
            token_ids = batch["input_ids"][i].cpu().tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

            baseline_pixels = torch.zeros_like(pixel_input).to(self.device)

            attributions = self.ig_audio.attribute(
                pixel_input,
                baselines=baseline_pixels,
                additional_forward_args=(input_ids[i:i+1], attention_mask[i:i+1]),
                n_steps=50
            )

            attr = attributions.squeeze().cpu().detach().numpy()
            attr = np.mean(np.abs(attr), axis=0)  # combine channels
            attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)

            orig = pixels[i].cpu().permute(1,2,0).numpy()
            orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)

            plt.figure(figsize=(6,6))
            plt.imshow(orig, aspect='auto')
            plt.imshow(attr, cmap='hot', alpha=0.5, aspect='auto')

            with torch.no_grad():
              pred = self.model(pixel_input, input_ids[i:i+1], attention_mask[i:i+1])
              pred_label = int(pred.item() >= 0.5)

            plt.title(f"Expected={labels[i]}, Predicted={pred_label}")
            plt.axis('off')
            plt.savefig(
              f"xai/fold{self.fold}_epoch{epoch}_sample{count}_true{labels[i]}_pred{pred_label}.png",
              bbox_inches="tight",
              pad_inches=0
            )
            plt.close()

            baseline_ids = torch.zeros_like(input_ids[i:i+1]).to(self.device)

            token_attr = self.lig_text.attribute(
                inputs=input_ids[i:i+1],
                baselines=baseline_ids,
                additional_forward_args=(pixel_input, attention_mask[i:i+1]),
                n_steps=50
            )

            token_attr = token_attr.sum(dim=-1).squeeze().cpu().detach().numpy()
            token_attr = (token_attr - token_attr.min()) / (token_attr.max() - token_attr.min() + 1e-8)

            token_scores = list(zip(tokens, token_attr))
            token_scores = sorted(token_scores, key=lambda x: x[1], reverse=True)

            with open(f"xai/fold{self.fold}_epoch{epoch}_tokens_sample{count}.csv", "w") as f:
              writer = csv.writer(f)
              writer.writerow(["token","importance"])
              writer.writerows(token_scores)
            
            count += 1

  def eval(self):
    self.model.eval() # Sets the model to evaluation mode. This disables dropout and ensures batch normalization uses running means/variances.
    label_probs = []
    label_true = [] # Initializes an empty list to store true labels.
    
    loader = self.val_loader # Uses the val data loader for evaluation. 

    with torch.no_grad(): # Disables gradient calculations. This is crucial during evaluation to save memory and computation, as we don't need to update model weights.
      for _, batch in enumerate(loader): # Iterates through batches from the val data loader.
        pixels = batch['pixels'].to(self.device) # Moves spectrograms to device.
        input_ids = batch["input_ids"].to(self.device) # Moves input IDs to device.
        attention_mask = batch["attention_mask"].to(self.device) # Moves attention mask to device.
        labels = batch['labels'] # Retrieves true labels kept on CPU initially.
        
        label_preds = self.model(pixels, input_ids, attention_mask) # Performs a forward pass to get predicted probabilities.
        
        label_probs += label_preds.detach().cpu().flatten().tolist()
        labels = labels.reshape(-1) # Reshapes true labels to 1D.
        label_true += labels.detach().to('cpu').tolist() # Detaches true labels, moves to CPU, and converts to a Python list.

      thresholds = np.linspace(0.05, 0.95, 50)
      best_f1 = 0
      best_preds = None

      for t in thresholds:
          preds = [1 if p >= t else 0 for p in label_probs]
          f1 = f1_score(label_true, preds, average='macro')

          if f1 > best_f1:
              best_f1 = f1
              self.best_threshold = t
              best_preds = preds

      print(f"Best Threshold: {self.best_threshold:.2f}")
        
    print(label_true, "true labels") 
    print(best_preds, "best preds")

    label_f1 = best_f1
    target_names = ['NonDementia', 'Dementia'] # Defines the names for the target classes for the classification report.
    print(classification_report(label_true, best_preds, target_names=target_names))
    roc = roc_auc_score(label_true, label_probs)
    print(f"ROC-AUC: {roc:.3f}")

    cm = confusion_matrix(label_true, best_preds)
    print("Confusion Matrix:\n", cm)
    return label_true, best_preds, label_f1, roc
    
def run_cv(args):
  cv_results = []
  dataset = CachedAdressoDataset(phase="all")
  labels = dataset.labels.cpu().numpy()
  groups = np.array(dataset.subject_ids)
  fold_scores = []

  sgkf = StratifiedGroupKFold(
      n_splits=args.n_folds,
      shuffle=True,
      random_state=args.seed
  )

  for fold, (train_idx, val_idx) in enumerate(
      sgkf.split(X=np.zeros(len(labels)), y=labels, groups=groups)
  ):
      print(f"\n========== Fold {fold+1}/{args.n_folds} ==========")

      torch.manual_seed(args.seed + fold)
      torch.cuda.manual_seed_all(args.seed + fold)
      np.random.seed(args.seed + fold)

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

      engine = Trainer(args, fold+1)
      engine.train_loader = train_loader
      engine.val_loader  = val_loader

      engine.train()

      print(f"Fold {fold+1} best threshold: {engine.best_threshold:.2f}")
      _, _, f1, roc_auc = engine.eval()

      fold_scores.append(f1)
      cv_results.append({
        "fold": fold+1,
        "f1": f1,
        "roc_auc": roc_auc,
        "threshold": engine.best_threshold
      })

  pd.DataFrame(cv_results).to_csv("cv_results.csv", index=False)
  print("\n===== Cross Validation Results =====")
  print(f"Mean F1: {np.mean(fold_scores):.3f}")
  print(f"Std F1: {np.std(fold_scores):.3f}")

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