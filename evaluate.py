import librosa
import yaml

import argparse

import attacks
from attacks.base import BaseAttack

parser = argparse.ArgumentParser()
parser.add_argument("config", help="YAML config file")
parser.add_argument("prefix", help="Prefix audio file")
parser.add_argument("log", nargs="?", default=None, help="Log output file")
args = parser.parse_args()

with open(args.config) as f:
    params = yaml.load(f, yaml.Loader)

print(f"\nEvaluating attack on {params['model_path']}\n")

prefix_audio, _ = librosa.load(args.prefix, sr=params["sr"])

attack: BaseAttack = params["attack"](params)

dataset = attack.load_dataset("test_dataset", False, False)

print("\nLoaded dataset")

attack.evaluate(prefix_audio, dataset, args.log)

print("\nEvaluation Complete\n")
