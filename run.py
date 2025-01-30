from src.utils.constants import SEED
import argparse
import random
import numpy as np
import torch
from src.preprocessing.preprocess import main as preprocess_main
from src.visualization.visualizer import main as visualization_main
from src.training.train import main as training_main
from src.evaluation.evaluate import main as evaluation_main
from src.utils.constants import MODEL_OPTIONS
from src.utils.logger import setup_logger

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Initialize logger
logger = setup_logger(name="pipeline", save_to_file=False)

# Map step names to their corresponding functions
STEP_FUNCTIONS = {
    "preprocess": preprocess_main,
    "visualize": visualization_main,
    "train": training_main,
    "evaluate": evaluation_main,
}

def choose_model():
    # Present model options to the user
    print("Please choose a model:")
    for idx, model in MODEL_OPTIONS.items():
        print(f"{idx}: {model}")

    # Get the user's choice and validate it
    while True:
        try:
            choice = int(input("Enter the number corresponding to your model choice: "))
            if choice in MODEL_OPTIONS:
                return MODEL_OPTIONS[choice]
            else:
                print("Invalid choice. Please select a valid model number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main(args):
    try:
        if args.steps == ["all"] or "train" in args.steps or "evaluate" in args.steps:
            model_name = choose_model()
        for step in args.steps:
            if step in STEP_FUNCTIONS:
                logger.info(f"Starting {step} step...")
                if step == "train" or step == "evaluate":
                    STEP_FUNCTIONS[step](model_name=model_name)
                else:
                    STEP_FUNCTIONS[step]()
                logger.info(f"{step.capitalize()} step completed.")

            elif step == "all":
                logger.info("Running all steps: preprocess, train, evaluate...")
                visualization_main()
                logger.info("Visualization step completed.")
                preprocess_main()
                logger.info("Preprocessing step completed.")
                training_main(model_name=model_name)
                logger.info("Training step completed.")
                evaluation_main(model_name=model_name)
                logger.info("Evaluation step completed.")
                break
            else:
                logger.error(f"Invalid step: {step}. Please choose from preprocess, train, evaluate, or all.")
    except Exception as e:
        logger.error(f"An error occurred during the {step} step: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different parts of the machine learning pipeline.")
    parser.add_argument(
        "steps",
        type=str,
        nargs="+",  # Accept one or more steps
        help="Specify the pipeline steps to execute: preprocess, train, evaluate, or all.",
    )
    args = parser.parse_args()
    main(args)
