import os
import yaml # type: ignore
import argparse
import logging
from typing import Text

from train import Train_Task

def main(config_path: Text) -> None:
    logging.basicConfig(level=logging.INFO)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    logging.info("Training started...")
    Train_Task(config).train()

    logging.info("Train complete")

    # logging.info("Evaluating...")
    # Test_Task(config).predict()
    # logging.info("Task done!!!")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    main(args.config)