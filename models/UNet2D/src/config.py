from pathlib import Path
from dotenv import dotenv_values, find_dotenv
import os

config = {
    **dotenv_values(find_dotenv(usecwd=True)),
    **os.environ
}

DATASET_PATH = Path(config["DATASET_PATH"])
SAVE_PATH = Path(config["SAVE_PATH"])

MODEL_VERSION = config["MODEL_VERSION"]
WEIGHTS_VERSION = config["WEIGHTS_VERSION"]

NUM_BRAINS = int(config["NUM_BRAINS"])
BATCH_SIZE = int(config["BATCH_SIZE"])

LR = float(config["LR"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])

STAGE1_EPOCHS = int(config.get("STAGE1_EPOCHS", 20))
STAGE2_EPOCHS = int(config.get("STAGE2_EPOCHS", 10))
STAGE3_EPOCHS = int(config.get("STAGE3_EPOCHS", 10))

CLASS_NAMES = [
    "background",
    "necrotic",
    "edema",
    "enhancing"
]