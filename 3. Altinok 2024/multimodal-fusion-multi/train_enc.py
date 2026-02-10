import torch
import torch.nn as nn
from torch.optim import AdamW
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
from bert_image import BertImage
from data_loader_vit import adresso_loader

class Trainer:
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        os.makedirs('saved_models', exist_ok=True) 
        os.makedirs('ckp', exist_ok=True)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        train_loader = adresso_loader(phase='train', batch_size=args.batch_size, shuffle=True)
        test_loader = adresso_loader(phase='test', batch_size=args.val_batch_size)

        if torch.cuda.device_count() > 0:
            print(f"{torch.cuda.device_count()} GPUs found")

        model = BertImage(nfinetune=args.nfinetune)
        model = nn.DataParallel(model)
        model.to(device)
        params = model.parameters()

        optimizer = AdamW(params, lr=args.lr, weight_decay=0.01)

        self.device = device
        self.model = model
        self.optimizer = optimizer
        
        # Define both loss functions
        self.binary_cross = nn.BCELoss()
        self.multi_class_loss = nn.CrossEntropyLoss()
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args
        self.epoch_accuracies = []
        self.all_losses = []
        
        # New list for combined losses
        self.all_combined_losses = [] 

    def train(self):
        best_acc = 0.0
        best_epoch = 0
        for epoch in range(self.args.epochs):
            print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
            loss, loss_mmse = self.train_epoch()
            
            # eval now returns two accuracies
            binary_acc, mmse_acc, _, _, _, _ = self.eval()
            print(f"Epoch {epoch + 1} Loss: {loss:.3f}") # Prints the average loss for the current epoch.
            print(f"Epoch {epoch + 1} MMSE Loss: {loss_mmse:.3f}") # Prints the average mmse loss for the current epoch.
            print(f'Binary Acc: {binary_acc:.3f}')
            print(f'MMSE Acc: {mmse_acc:.3f}')
            
            combined_acc = (binary_acc + mmse_acc) / 2
            self.epoch_accuracies.append(round(combined_acc, 3))

            if combined_acc > best_acc:
                best_acc = combined_acc
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), f'saved_models/best_model.pth')
                print(f"Best model saved at epoch {best_epoch} with accuracy {best_acc:.3f}")
            
        with open("epoch_acc.txt", "w") as ofile:
            for eacc in self.epoch_accuracies:
                ofile.write(str(eacc) + "\n")
        
        # Save the combined losses
        with open("losses.txt", "w") as ofile:
            for eloss in self.all_combined_losses:
                ofile.write(str(eloss) + "\n")

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        epoch_loss_mmse = 0
        
        for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            self.optimizer.zero_grad()
            
            pixels = batch['pixels'].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Get both sets of labels from the batch
            labels = batch["labels"].to(self.device).reshape(-1)
            mmse_scores = batch["mmse_scores"].to(self.device)
            
            # Perform a single forward pass
            binary_prob, mmse_logits = self.model(pixels, input_ids, attention_mask)
            
            # Calculate both losses
            loss_binary = self.binary_cross(binary_prob.reshape(-1), labels.float())
            loss_mmse = self.multi_class_loss(mmse_logits, mmse_scores)
            
            # Combine the losses and perform one backward pass
            total_loss = loss_binary + loss_mmse
            total_loss.backward()
            self.optimizer.step()
            
            # Log the combined loss
            interval = max(len(self.train_loader) // 20, 1)
            if i % interval == 0 or i == len(self.train_loader) - 1:
                lloss = round(total_loss.item(), 3)
                print(f'Batch: {i + 1}/{len(self.train_loader)}\ttotal combined loss: {lloss:.3f}')
                self.all_combined_losses.append(lloss)

            epoch_loss += loss_binary.item()
            epoch_loss_mmse += loss_mmse.item()
            
        return epoch_loss / len(self.train_loader), epoch_loss_mmse / len(self.train_loader)

    def eval(self):
        self.model.eval()
        label_pred = []
        label_true = []
        mmse_pred = []
        mmse_true = []
        
        loader = self.test_loader

        with torch.no_grad():
            for _, batch in enumerate(loader):
                pixels = batch['pixels'].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Get both sets of labels from the batch
                labels = batch['labels'].reshape(-1)
                mmse_scores = batch['mmse_scores']
                
                # Perform a single forward pass
                binary_prob, mmse_logits = self.model(pixels, input_ids, attention_mask)
                
                # Process binary predictions
                label_pred += binary_prob.detach().to('cpu').flatten().round().long().tolist()
                label_true += labels.detach().to('cpu').tolist()
                
                # Process MMSE predictions
                predicted_classes = torch.argmax(mmse_logits, dim=1)
                mmse_pred += predicted_classes.detach().to('cpu').tolist()
                mmse_true += mmse_scores.detach().to('cpu').tolist()
                
        # Calculate and print metrics for both tasks
        label_acc = accuracy_score(label_true, label_pred)
        mmse_acc = accuracy_score(mmse_true, mmse_pred)

        print("\n[Binary Classification Report]")
        target_names_binary = ['NonDementia', 'Dementia']
        print(classification_report(label_true, label_pred, target_names=target_names_binary))
        
        print("\n[MMSE Classification Report]")
        target_names_mmse = ['Severe (0-9)', 'Moderate (10-18)', 'Mild (19-23)', 'Normal (24-30)']
        print(classification_report(mmse_true, mmse_pred, target_names=target_names_mmse, labels=[0, 1, 2, 3], zero_division=0))
        
        return label_acc, mmse_acc, label_true, label_pred, mmse_true, mmse_pred
    
    def test(self):
        self.model.load_state_dict(torch.load('saved_models/best_model.pth'))
        final_acc_binary, final_acc_mmse, label_true, label_pred, mmse_true, mmse_pred = self.eval()
        cm_binary = confusion_matrix(label_true, label_pred)
        print("Confusion Matrix for Binary Classification:\n", cm_binary)
        cm_mmse = confusion_matrix(mmse_true, mmse_pred)
        print("Confusion Matrix for MMSE Classification:\n", cm_mmse)
        print(f"Final Test Accuracy for Label: {final_acc_binary:.3f}")
        print(f"Final Test Accuracy for MMSE: {final_acc_mmse:.3f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of training')
    parser.add_argument('--val_batch_size', type=int, default=32, help='batch size of testing')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--gpu', type=str, default='', help='GPUs to use')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--nfinetune', type=int, default=48)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    engine = Trainer(args)
    engine.train()
    engine.test()