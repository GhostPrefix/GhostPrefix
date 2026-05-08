import yaml

import argparse

import attacks
from attacks.base import BaseAttack

parser = argparse.ArgumentParser()
parser.add_argument("config", help="YAML config file")
args = parser.parse_args()

with open(args.config) as f:
    params = yaml.load(f, yaml.Loader)

print(f"\nAttacking {params['model_path']}\n")

attack: BaseAttack = params["attack"](params)

train_dataset = attack.load_dataset("train_dataset")
eval_dataset = attack.load_dataset("test_dataset", False, False)

print(f"\nLoaded datasets")

attack.train(train_dataset, eval_dataset)

print("\nTraining Complete\n")
