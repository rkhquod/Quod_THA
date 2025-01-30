from pathlib import Path

SEED = 42

# Define configuration file paths
CONFIG_PREPROCESS = Path("configs/preprocess_config.yaml")
CONFIG_TRAIN = Path("configs/train_config.yaml")
CONFIG_EVAL = Path("configs/evaluation_config.yaml")
CONFIG_MODEL = Path("configs/model_config.yaml")

# Shared project constants
TARGET_COLUMN = "transactions"

# Viz folder
OUTPUT_FOLDER = Path("outputs/")

RAW_DATA_PATHS = [
    Path("data/raw/transactions_1.csv"),
    Path("data/raw/transactions_2.csv"),
]

MODEL_OPTIONS = {
    1: "NeuralNet",
    2: "XGBoost",
    3: "RandomForest",
    4: "RidgeML"
}