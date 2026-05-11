from datasets import Audio, Dataset, concatenate_datasets, load_dataset
import jiwer
import librosa
import numpy as np
import soundfile
import torch

from collections.abc import Callable
import json
import logging
import os
import random

from models import OmniAsrCtcAttack, OmniAsrCtcAttackOnHiddenStates, OmniAsrLlmAttack, OmniAsrLlmAttackOnHiddenStates, Wav2Vec2Attack, Wav2Vec2AttackOnHiddenStates, WhisperAttack, WhisperAttackOnHiddenStates
from models.base import BaseModelAttack, BaseModelAttackOnHiddenStates
from .robust import RobustPipeline


def dataset_map(example):
    example["audio"] = example["audio"]["array"]
    if "transcript" in example:
        example["text"] = example.pop("transcript")

    return example


class BaseAttack():
    hidden_states = False
    use_transcriptions_cache = False

    def __init__(self, params: dict):
        self.params = params

        self.rng = np.random.default_rng(self.params["seed"])
        torch.manual_seed(self.params["seed"])
        random.seed(self.params["seed"])

        self.robust = RobustPipeline(self)

        self.model: BaseModelAttack | BaseModelAttackOnHiddenStates | None = None
        if self.hidden_states:
            self.load_hidden_states_model()
        else:
            self.load_model()

        if self.use_transcriptions_cache:
            self.init_transcriptions_cache()

    def load_model(self):
        if "omniASR_CTC" in self.params["model_path"]:
            self.model = OmniAsrCtcAttack(self.params, self.robust)
        elif "omniASR_LLM" in self.params["model_path"]:
            self.model = OmniAsrLlmAttack(self.params, self.robust)
        elif "wav2vec2" in self.params["model_path"]:
            self.model = Wav2Vec2Attack(self.params, self.robust)
        elif "whisper" in self.params["model_path"]:
            self.model = WhisperAttack(self.params, self.robust)
        else:
            raise NotImplementedError

    def load_hidden_states_model(self):
        if "omniASR_CTC" in self.params["model_path"]:
            self.model = OmniAsrCtcAttackOnHiddenStates(self.params, self.robust)
        elif "omniASR_LLM" in self.params["model_path"]:
            self.model = OmniAsrLlmAttackOnHiddenStates(self.params, self.robust)
        elif "wav2vec2" in self.params["model_path"]:
            self.model = Wav2Vec2AttackOnHiddenStates(self.params, self.robust)
        elif "whisper" in self.params["model_path"]:
            self.model = WhisperAttackOnHiddenStates(self.params, self.robust)
        else:
            raise NotImplementedError

    def init_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.params["log_level"])

        self.save_dir: str = os.path.join(self.params["root"], self.params["save_folder"], self.params["model_path"].split("/")[-1])
        self.optim_save_dir = os.path.join(self.save_dir, "optim")

        os.makedirs(self.optim_save_dir, exist_ok=True)

        log_file = os.path.join(self.save_dir, "train.log")
        self.logger.addHandler(logging.FileHandler(log_file))

    def init_transcriptions_cache(self):
        self.transcriptions_cache_dir = os.path.join(self.params["cache"], "base_transcriptions", self.params["train_dataset"]["path"])
        self.transcriptions_cache_file = os.path.join(self.transcriptions_cache_dir, f"{self.params['model_path'].replace('/', '-')}.json")

        if os.path.exists(self.transcriptions_cache_file):
            with open(self.transcriptions_cache_file) as f:
                self.transcriptions_cache: dict = json.load(f)
        else:
            self.transcriptions_cache = {}

    def init_prefix(self) -> torch.Tensor:
        if self.params["prefix_ckpt"]:
            prefix_audio, _ = librosa.load(os.path.join(self.save_dir, f"prefix-{self.params['prefix_ckpt']}.wav"), sr=self.params["sr"])
        elif self.params["initial_prefix"]:
            prefix_audio, _ = librosa.load(self.params["initial_prefix"], sr=self.params["sr"])
        else:
            prefix_audio = self.rng.uniform(-self.params["epsilon"], self.params["epsilon"], int(self.params["prefix_length"] * self.params["sr"]))

        prefix_input_values = self.model.preprocess_for_attack(prefix_audio)
        prefix_input_values.requires_grad_()

        if self.params["initial_prefix"]:
            init_prefix_audio, _ = librosa.load(self.params["initial_prefix"], sr=self.params["sr"])
            self.model.initial_prefix_inputs = self.model.preprocess_for_attack(init_prefix_audio)

        self.model.optimizer = self.params["optimizer"]([prefix_input_values], lr=self.params["alpha"])
        if self.params["prefix_ckpt"]:
            self.model.optimizer.load_state_dict(torch.load(os.path.join(self.optim_save_dir, f"prefix-{self.params['prefix_ckpt']}.pt")))

        return prefix_input_values

    def save_prefix(self, epoch: int, prefix_audio: np.ndarray) -> str:
        save_file = os.path.join(self.save_dir, f"prefix-{epoch}.wav")

        soundfile.write(save_file, prefix_audio, self.params["sr"])
        torch.save(self.model.optimizer.state_dict(), os.path.join(self.optim_save_dir, f"prefix-{epoch}.pt"))

        return save_file

    def load_dataset(self, key: str, shuffle: bool = True, batch: bool = True) -> Dataset:
        cache_dir = os.path.join(self.params["cache"], "huggingface", "datasets")
        splits = self.params[key].get("splits")
        name = self.params[key].get("name")

        dataset_list = []
        if splits:
            for s in splits:
                dataset_list.append(load_dataset(self.params[key]["path"], name, split=s, cache_dir=cache_dir, streaming=True))
        else:
            dataset_list.append(load_dataset(self.params[key]["path"], name, cache_dir=cache_dir, streaming=True))

        dataset = concatenate_datasets(dataset_list)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.params["sr"]))
        dataset = dataset.map(dataset_map)

        if shuffle:
            dataset = dataset.shuffle(self.params["seed"], self.rng, 10000)
        if batch:
            dataset = dataset.batch(self.params["batch_size"], drop_last_batch=True)

        return dataset

    def inputs_to_audio(self, input_values: torch.Tensor) -> np.ndarray:
        return input_values.squeeze().detach().cpu().float().numpy(force=True)

    def standardize_text(self, text: str | list[str]) -> str | list[str]:
        transform = jiwer.Compose([jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ToUpperCase()])
        punctuation = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~"  # Allow '

        text = transform(text)
        if isinstance(text, list):
            text = ["".join(c for c in t if c not in punctuation) for t in text]
        else:
            text = "".join(c for c in text if c not in punctuation)

        return text

    def get_results(self, reference: str | list[str], hypothesis: str | list[str], success: Callable[[float], bool], standardize: bool = True) -> tuple[float, bool]:
        if standardize:
            reference, hypothesis = self.standardize_text(reference), self.standardize_text(hypothesis)

        wer = jiwer.wer(reference, hypothesis)
        return wer, success(wer)

    def get_base_inputs_and_transcriptions(self, batch) -> tuple[torch.Tensor, list[str]]:
        base_input_values = self.model.preprocess_for_attack(batch["audio"])

        base_transcriptions = []
        for sample_id in batch["id"]:
            sample_id = str(sample_id)
            if sample_id in self.transcriptions_cache:
                base_transcriptions.append(self.transcriptions_cache[sample_id])
            else:
                base_transcriptions = self.model.infer_batch(base_input_values)
                for i, tr in zip(batch["id"], base_transcriptions):
                    self.transcriptions_cache[str(i)] = tr
                break

        return base_input_values, base_transcriptions

    def cache_base_transcriptions(self):
        os.makedirs(self.transcriptions_cache_dir, exist_ok=True)
        with open(self.transcriptions_cache_file, "w") as f:
            json.dump(self.transcriptions_cache, f, indent=4)

    def train(self, train_dataset: Dataset, eval_dataset: Dataset | None):
        pass

    def evaluate(self, prefix_audio: np.ndarray, dataset: Dataset, log_file: str | None = None, print_output: bool = True, eval_no_atk: bool = True):
        pass
