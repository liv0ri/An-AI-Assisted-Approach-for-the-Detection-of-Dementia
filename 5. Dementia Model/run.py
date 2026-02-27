import subprocess
from config import BATCH_SIZE, VAL_BATCH_SIZE, EPOCHS, GPUS, LR, NFINETUNE, N_FOLDS

command = [
    "python", "train_enc.py",
    "--gpu", GPUS,
    "--batch_size", str(BATCH_SIZE),
    "--val_batch_size", str(VAL_BATCH_SIZE),
    "--epochs", str(EPOCHS),
    "--lr", str(LR),
    "--nfinetune", str(NFINETUNE),
    "--n_folds", str(N_FOLDS)
]

subprocess.run(command, check=True)