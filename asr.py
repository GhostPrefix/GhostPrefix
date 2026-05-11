import librosa
import yaml

import argparse

import attacks
from attacks.base import BaseAttack

parser = argparse.ArgumentParser()
parser.add_argument("config", help="YAML config file")
parser.add_argument("audio", help="Audio file")
args = parser.parse_args()

with open(args.config) as f:
    params = yaml.load(f, yaml.Loader)

print(f"\nPerforming ASR on '{args.audio}' with {params['model_path']}\n")

audio_input, _ = librosa.load(args.audio, sr=params["sr"])

attack: BaseAttack = params["attack"](params)

transcription = attack.model.infer_single(audio_input)

print(f"\nTranscription: {transcription}\n")
