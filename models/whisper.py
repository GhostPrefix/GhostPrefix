import numpy as np
import torch
from whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, N_FRAMES
from whisper.tokenizer import get_tokenizer
import whisper

import os

from .base import BaseModelAttack, BaseModelAttackOnHiddenStates
from .whisper_loss import get_loss_from_mel


class WhisperAttack(BaseModelAttack):
    def __init__(self, params, robust):
        super().__init__(params, robust)

        self.model = whisper.load_model(self.params["model_path"].replace("openai/whisper-", ""), download_root=os.path.join(self.params["cache"], "whisper"))
        self.tokenizer = get_tokenizer(self.model.is_multilingual)

        self.model.eval()
        self.model.to(self.device)

        self.confidence = 0.0
        self.correct_first_word = True

        self.target_ids: torch.Tensor | None = None

    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        mel = log_mel_spectrogram(audio, self.model.dims.n_mels, padding=N_SAMPLES)
        mel = pad_or_trim(mel, N_FRAMES)
        return mel

    def transform_text(self, text: str) -> str:
        return text  # Pass through

    def get_targets(self, transcriptions: list[str]):
        encoded = [self.tokenizer.encode(t, disallowed_special=()) for t in transcriptions]
        max_len = max(len(row) for row in encoded)

        padded_ids = np.array([np.pad(row, (0, max_len - len(row)), constant_values=self.tokenizer.eot) for row in encoded], dtype=np.int64)
        self.target_ids = torch.from_numpy(padded_ids)

    def preprocess_for_attack(self, audio: np.ndarray | list | tuple) -> torch.Tensor:
        # If batched
        if (isinstance(audio, np.ndarray) and len(audio.shape) > 1) or (isinstance(audio, (list, tuple)) and (isinstance(audio[0], (np.ndarray, tuple, list)))):
            max_length = max(len(a) for a in audio)
            padded_audio = [pad_or_trim(np.array(a), max_length) for a in audio]
            inputs = np.array(padded_audio, dtype=np.float32)
        else:
            inputs = np.array([audio], dtype=np.float32)

        inputs = torch.from_numpy(inputs)
        inputs = inputs.to(self.device)

        return inputs

    def infer_batch(self, input_values: torch.Tensor) -> list[str]:
        with torch.no_grad():
            mel = self.audio_to_mel(input_values)
            results = self.model.decode(mel)

        return [r.text for r in results]

    def infer_batch_with_prefix(self, prefix_input_values: torch.Tensor, base_input_values: torch.Tensor) -> list[str]:
        concat_input_values = self.prepend_to_base_inputs(prefix_input_values, base_input_values)

        return [self.infer_single(x) for x in concat_input_values]

    def infer_single(self, audio_input: np.ndarray) -> str:
        with torch.no_grad():
            return self.model.transcribe(audio_input)["text"].strip()

    def attack_step(self, prefix_inputs: torch.Tensor, base_inputs: torch.Tensor, neg: bool = False) -> torch.Tensor:
        concat_input_values = self.prepend_to_base_inputs(prefix_inputs, base_inputs)

        # Forward pass with prefix || base audios
        mel = self.audio_to_mel(concat_input_values)
        results = get_loss_from_mel(self.model, mel, self.target_ids, self.confidence, self.correct_first_word)

        # Compute mean model loss
        loss = torch.stack(tuple(r.loss for r in results)).mean()

        return self.finish_attack_step(loss, prefix_inputs, neg)


class WhisperAttackOnHiddenStates(BaseModelAttackOnHiddenStates, WhisperAttack):

    def get_hidden_states(self, audio_inputs: torch.Tensor) -> torch.Tensor:
        mel = self.audio_to_mel(audio_inputs)
        hidden_states = self.model.encoder(mel)
        return hidden_states

    def get_truth(self, original_inputs: torch.Tensor):
        with torch.no_grad():
            self.true_hidden_states = self.get_hidden_states(original_inputs)

    def attack_step(self, prefix_inputs: torch.Tensor, base_inputs: torch.Tensor, neg: bool = False) -> torch.Tensor:
        # Get true hidden states
        self.get_truth(base_inputs)

        concat_input_values = self.prepend_to_base_inputs(prefix_inputs, base_inputs)

        # Forward pass with prefix || base audios and get hidden states
        adv_hidden_states = self.get_hidden_states(concat_input_values)

        # Compute hidden model loss
        hidden_loss = self.loss(adv_hidden_states, self.true_hidden_states)

        return self.finish_attack_step(hidden_loss, prefix_inputs, neg)
