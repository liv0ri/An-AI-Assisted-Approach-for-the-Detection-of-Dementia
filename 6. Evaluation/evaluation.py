import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, average_precision_score
import seaborn as sns
from tqdm import tqdm
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "5. Dementia Model")

sys.path.append(MODEL_DIR)
from cached_adresso_dataset import adresso_loader
from bert_image import BertImage

# Configuration
MODEL_PATH = 'saved_models/best_model.pth'
MODEL_PATH = os.path.join('5. Dementia Model', MODEL_PATH)
ROBERTA_MODEL_NAME = "FacebookAI/roberta-base"
BATCH_SIZE = 32
# expriment with different threshold values to optimize performance IMPOOO
PROBABILITY_THRESHOLD = 0.5
CONF_MATRIX_FILE = 'confusion_matrix.png'
ROC_CURVE_FILE = 'roc_curve.png'
PR_CURVE_FILE = 'precision_recall_curve.png'
save_dir = r"D:\Uni\thesis final\Other\b. Diagram of Evaluation"

class ModelEvaluator:
    def __init__(self, model_path=MODEL_PATH, batch_size=BATCH_SIZE):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.batch_size = batch_size
        self.model = None
        os.makedirs(save_dir, exist_ok=True)

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        print(f"Loading model from {self.model_path}...")
        
        # Initialize model structure (nfinetune doesn't matter for evaluation as we load weights)
        self.model = BertImage(nfinetune=0) 
        
        # Load state dictionary
        state_dict = torch.load(self.model_path, map_location=self.device)
        
        # Handle DataParallel prefix if present (remove 'module.')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def evaluate(self, phase='test'):
        print(f"Starting evaluation on '{phase}' set...")
        
        loader = adresso_loader(phase=phase, batch_size=self.batch_size, base_path="5. Dementia Model")
        
        y_true = []
        y_probs = []
        records = []  
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                pixels = batch['pixels'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].cpu().numpy()
                
                transcripts = batch.get('raw_texts', [''] * len(labels))
                file_names = batch.get('file_names', ['unknown'] * len(labels))

                outputs = self.model(pixels, input_ids, attention_mask)
                probs = outputs.squeeze().cpu().numpy()

                for i in range(len(probs)):
                    prob = float(probs[i])
                    true_label = int(labels[i])
                    pred_label = int(prob >= PROBABILITY_THRESHOLD)

                    y_probs.append(prob)
                    y_true.append(true_label)

                    records.append({
                        "file_name": file_names[i],
                        "transcript": transcripts[i],
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "probability": round(prob, 4),
                        "correct": pred_label == true_label
                    })

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(save_dir, "evaluation_details.csv"), index=False)
        print("Saved evaluation_details.csv")
        if pred_label != true_label:
            img = pixels[i].cpu().permute(1, 2, 0).numpy()
            plt.imsave(
                os.path.join(save_dir, f"misclassified_{file_names[i]}"),
                img,
                cmap="viridis"
            )


        return np.array(y_true), np.array(y_probs)


    def plot_confusion_matrix(self, y_true, y_pred, save_path=CONF_MATRIX_FILE):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Control', 'Dementia'], 
                    yticklabels=['Control', 'Dementia'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, save_path),
            dpi=300,
            bbox_inches="tight"
        )
        print(f"Confusion matrix saved to {save_path}")
        plt.close()

    def plot_roc_curve(self, y_true, y_probs, save_path=ROC_CURVE_FILE):
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, save_path),
            dpi=300,
            bbox_inches="tight"
        )
        print(f"ROC curve saved to {save_path}")
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_probs, save_path=PR_CURVE_FILE):
        val_precision, val_recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(val_recall, val_precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(val_recall, val_precision, color='blue', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, save_path),
            dpi=300,
            bbox_inches="tight"
        )
        print(f"Precision-Recall curve saved to {save_path}")
        plt.close()

if __name__ == "__main__":
    # Initialize and run evaluation
    evaluator = ModelEvaluator()
    evaluator.load_model()
    
    # Get predictions
    y_true, y_probs = evaluator.evaluate(phase='test')
    
    # Convert probabilities to binary predictions
    y_pred = (y_probs >= PROBABILITY_THRESHOLD).astype(int)
    
    # Print metrics
    print("\n" + "="*80)
    print("       Evaluation Report")
    print("="*80)
    print(classification_report(y_true, y_pred, target_names=['Control', 'Dementia']))

    # Macro F1
    print("\n" + "="*80)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    # Generate Plots
    print("\nGenerating plots...")
    evaluator.plot_confusion_matrix(y_true, y_pred)
    evaluator.plot_roc_curve(y_true, y_probs)
    evaluator.plot_precision_recall_curve(y_true, y_probs)
    print("\nDone! Plots saved locally.")
