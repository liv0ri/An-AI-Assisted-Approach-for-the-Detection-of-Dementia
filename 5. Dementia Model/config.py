BATCH_SIZE = 64
VAL_BATCH_SIZE = 32
EPOCHS = 30
GPUS = "0"
LR = 3e-5
NFINETUNE = 8
N_FOLDS = 5

OUTPUT_DIR = "diagnosis"
MAIN_FOLDER = "5. Dementia Model"
MODALITIES = ["wav", "specto", "trans"]
ORIGINAL_CLASSES = ["control", "dementia"]
CLASS_MAP = {"control": "cn", "dementia": "ad"}
TEST_CSV_NAME = "task1.csv"