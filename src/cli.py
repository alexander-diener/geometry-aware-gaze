import argparse
from src.learning.train import main as train_main

def parse_args():
    p = argparse.ArgumentParser("geometry-aware-gaze")
    p.add_argument("--mode", choices=["train"], default="train")
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "train":
        train_main()

if __name__ == "__main__":
    main()