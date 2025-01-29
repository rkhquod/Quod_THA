import argparse
import logging
from src.preprocessing.preprocess import main as preprocess_main
# from src.visualization.visualizer import main as visualization_main
from src.training.train import main as training_main
from src.evaluation.evaluate import main as evaluation_main

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Map step names to their corresponding functions
STEP_FUNCTIONS = {
    "preprocess": preprocess_main,
    # "visualize": visualization_main,
    "train": training_main,
    "evaluate": evaluation_main,
}

def main(args):
    try:
        for step in args.steps:
            if step in STEP_FUNCTIONS:
                logging.info(f"Starting {step} step...")
                STEP_FUNCTIONS[step]()  # Execute the step
                logging.info(f"{step.capitalize()} step completed.")
            elif step == "all":
                logging.info("Running all steps: preprocess, train, evaluate...")
                preprocess_main()
                logging.info("Preprocessing step completed.")
                training_main()
                logging.info("Training step completed.")
                evaluation_main()
                logging.info("Evaluation step completed.")
                break  
            else:
                logging.error(f"Invalid step: {step}. Please choose from preprocess, train, evaluate, or all.")
    except Exception as e:
        logging.error(f"An error occurred during the {step} step: {e}")
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