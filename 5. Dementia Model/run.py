import subprocess
import sys
import os
from config import (
    BATCH_SIZE,
    VAL_BATCH_SIZE,
    EPOCHS,
    GPUS,
    LR,
    NFINETUNE,
    N_FOLDS,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

command = [
    sys.executable,            
    "train_enc.py",
    "--gpu", GPUS,
    "--batch_size", str(BATCH_SIZE),
    "--val_batch_size", str(VAL_BATCH_SIZE),
    "--epochs", str(EPOCHS),
    "--lr", str(LR),
    "--nfinetune", str(NFINETUNE),
    "--n_folds", str(N_FOLDS),
]

subprocess.run(
    command,
    check=True,
    cwd=THIS_DIR                 
)