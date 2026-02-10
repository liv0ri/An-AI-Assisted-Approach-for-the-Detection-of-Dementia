import subprocess

BATCH_SIZE = 64
VAL_BATCH_SIZE = 32
EPOCHS = 30
GPUS = "0"
LR = 3e-5
NFINETUNE = 8

command = [
    "python", "train_enc.py",
    "--gpu", GPUS,
    "--batch_size", str(BATCH_SIZE),
    "--val_batch_size", str(VAL_BATCH_SIZE),
    "--epochs", str(EPOCHS),
    "--lr", str(LR),
    "--nfinetune", str(NFINETUNE)
]

subprocess.run(command, check=True)