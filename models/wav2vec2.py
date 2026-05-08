import jiwer
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers.feature_extraction_utils import BatchFeature

import os

from .base import BaseModelAttack, BaseModelAttackOnHiddenStates


class Wav2Vec2Attack(BaseModelAttack):
    def __init__(self, params, robust):
        super().__init__(params, robust)

        self.processor = Wav2Vec2Processor.from_pretrained(self.params["model_path"], cache_dir=os.path.join(self.params["cache"], "huggingface", "hub"))
        self.model = Wav2Vec2ForCTC.from_pretrained(self.params["model_path"], cache_dir=os.path.join(self.params["cache"], "huggingface", "hub"))

        self.model.eval()
        self.model.to(self.device)

        self.loss = nn.CTCLoss(self.processor.tokenizer.pad_token_id)

        self.target_ids: torch.Tensor | None = None
        self.target_lengths: torch.Tensor | None = None

    def transform_text(self, text: str) -> str:
        transform = jiwer.Compose([jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ToUpperCase()])
        vocab = self.processor.tokenizer.get_vocab()
        vocab[" "] = vocab.pop("|")

        text = transform(text)
        text = "".join(c for c in text if c in vocab)

        return text

    def get_targets(self, transcriptions: list[str]):
        self.target_ids = self.processor.tokenizer(transcriptions, padding=True, return_tensors="pt").input_ids.to(self.device)
        self.target_lengths = torch.full((self.target_ids.shape[0],), self.target_ids.shape[1])

    def preprocess_for_attack(self, audio: np.ndarray | list | tuple) -> torch.Tensor:
        # If batched
        if (isinstance(audio, np.ndarray) and len(audio.shape) > 1) or (isinstance(audio, (list, tuple)) and (isinstance(audio[0], (np.ndarray, tuple, list)))):
            inputs = BatchFeature({"input_values": audio})
            inputs = self.processor.pad(inputs, padding=True)
        else:
            inputs = BatchFeature({"input_values": [audio]})

        inputs["input_values"] = [array.astype(np.float32) for array in inputs["input_values"]]
        inputs = inputs.convert_to_tensors("pt")
        inputs = inputs.to(self.device)

        return inputs.input_values

    def infer_batch(self, input_values: torch.Tensor) -> list[str]:
        with torch.no_grad():
            outputs = self.model(input_values=input_values)
            transcriptions = self.processor.batch_decode(torch.argmax(outputs.logits, dim=-1))

        return transcriptions

    def infer_batch_with_prefix(self, prefix_input_values: torch.Tensor, base_input_values: torch.Tensor) -> list[str]:
        concat_input_values = self.prepend_to_base_inputs(prefix_input_values, base_input_values)
        concat_input_values = self.normalize_audio_inputs(concat_input_values)

        with torch.no_grad():
            outputs = self.model(input_values=concat_input_values)
            transcriptions = self.processor.batch_decode(torch.argmax(outputs.logits, dim=-1))

        return transcriptions

    def infer_single(self, audio_input: np.ndarray) -> str:
        # Preprocess the audio input
        input_values = self.processor(audio_input, sampling_rate=self.params["sr"], return_tensors="pt").input_values
        input_values = input_values.to(self.device)

        # Infer
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Decode the predicted IDs to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        return transcription

    def attack_step(self, prefix_inputs: torch.Tensor, base_inputs: torch.Tensor, neg: bool = False) -> torch.Tensor:
        concat_input_values = self.prepend_to_base_inputs(prefix_inputs, base_inputs)

        # Forward pass with prefix || base audios
        outputs = self.model(input_values=concat_input_values)

        log_probs = nn.functional.log_softmax(outputs.logits, dim=-1)
        lengths = torch.full((log_probs.shape[0],), log_probs.shape[1])

        # Compute loss
        cost = self.loss(log_probs.transpose(0, 1), self.target_ids, lengths, self.target_lengths)

        return self.finish_attack_step(cost, prefix_inputs, neg)


class Wav2Vec2AttackOnHiddenStates(BaseModelAttackOnHiddenStates, Wav2Vec2Attack):

    def get_truth(self, original_inputs: torch.Tensor):
        with torch.no_grad():
            original_outputs = self.model(input_values=original_inputs, labels=None, output_hidden_states=True)
            self.true_hidden_states = original_outputs.hidden_states[-1]

    def attack_step(self, prefix_inputs: torch.Tensor, base_inputs: torch.Tensor, neg: bool = False) -> torch.Tensor:
        concat_input_values = self.prepend_to_base_inputs(prefix_inputs, base_inputs)

        # Get true hidden states
        padded_base_inputs = torch.nn.functional.pad(base_inputs, (0, concat_input_values.shape[1] - base_inputs.shape[1]))
        self.get_truth(padded_base_inputs)

        # Forward pass with prefix || base audios
        outputs = self.model(input_values=concat_input_values, labels=None, output_hidden_states=True)
        adv_hidden_states = outputs.hidden_states[-1]  # Last hidden layer for adversarial input

        # Compute hidden loss
        hidden_loss = self.loss(adv_hidden_states, self.true_hidden_states)

        return self.finish_attack_step(hidden_loss, prefix_inputs, neg)
