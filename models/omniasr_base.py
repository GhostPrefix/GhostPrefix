from fairseq2.nn.batch_layout import BatchLayout
import jiwer
import numpy as np
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import torch
from whisper.audio import pad_or_trim

import os
import string

from .base import BaseModelAttack, BaseModelAttackOnHiddenStates


class OmniAsrBaseAttack(BaseModelAttack):
    def __init__(self, params, robust):
        super().__init__(params, robust)

        os.environ["FAIRSEQ2_CACHE_DIR"] = os.path.join(self.params["cache"], "fairseq2", "assets")

        self.pipeline = ASRInferencePipeline(self.params["model_path"].replace("facebook/", ""), device=self.device)

        self.target_ids: torch.Tensor | None = None
        self.target_layout: BatchLayout | None = None

        self.lang = None

    def get_batch_layout(self, input_values: torch.Tensor) -> BatchLayout:
        # Get lengths of audios minus padding
        tz = (input_values.flip(1) == 0.0).cumprod(1).sum(1)
        seq_lens = [input_values.shape[1] - i for i in tz]

        return BatchLayout(input_values.shape, seq_lens=seq_lens, device=self.device)

    def transform_text(self, text: str) -> str:
        transform = jiwer.Compose([jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ToLowerCase()])
        vocab = string.ascii_lowercase + " '"

        text = transform(text)
        text = "".join(c for c in text if c in vocab)

        return text

    def get_targets(self, transcriptions: list[str]):
        encoded = [self.pipeline.token_encoder(t) for t in transcriptions]
        collated = self.pipeline.collater_text(encoded)

        self.target_ids = collated["seqs"].to(self.device)
        self.target_layout = BatchLayout(collated["seqs"].shape, seq_lens=collated["seq_lens"])

    def preprocess_for_attack(self, audio: np.ndarray | list | tuple) -> torch.Tensor:
        # If batched
        if (isinstance(audio, np.ndarray) and len(audio.shape) > 1) or (isinstance(audio, (list, tuple)) and (isinstance(audio[0], (np.ndarray, tuple, list)))):
            max_length = max(len(a) for a in audio)
            padded_audio = [pad_or_trim(np.array(a), max_length) for a in audio]
            inputs = np.array(padded_audio, dtype=np.float32)
        else:
            inputs = np.array([audio], dtype=np.float32)

        inputs = torch.from_numpy(inputs)
        inputs = inputs.to(self.device, torch.bfloat16)

        return inputs

    def infer_batch_with_prefix(self, prefix_input_values: torch.Tensor, base_input_values: torch.Tensor) -> list[str]:
        concat_input_values = self.prepend_to_base_inputs(prefix_input_values, base_input_values)

        inp = [{"waveform": arr, "sample_rate": self.params["sr"]} for arr in concat_input_values.float().numpy(force=True)]
        with torch.no_grad():
            return self.pipeline.transcribe(inp, lang=[self.lang] * len(inp), batch_size=len(inp))

    def infer_single(self, audio_input: np.ndarray) -> str:
        inp = [{"waveform": audio_input, "sample_rate": self.params["sr"]}]

        with torch.no_grad():
            transcriptions = self.pipeline.transcribe(inp, lang=[self.lang], batch_size=1)

        return transcriptions[0]


class OmniAsrBaseAttackOnHiddenStates(BaseModelAttackOnHiddenStates, OmniAsrBaseAttack):

    def get_hidden_states(self, audio_inputs: torch.Tensor) -> torch.Tensor:
        pass

    def get_truth(self, original_inputs: torch.Tensor):
        with torch.no_grad():
            self.true_hidden_states = self.get_hidden_states(original_inputs)

    def attack_step(self, prefix_inputs: torch.Tensor, base_inputs: torch.Tensor, neg: bool = False) -> torch.Tensor:
        concat_input_values = self.prepend_to_base_inputs(prefix_inputs, base_inputs)

        # Get true hidden states
        padded_base_inputs = torch.nn.functional.pad(base_inputs, (0, concat_input_values.shape[1] - base_inputs.shape[1]))
        self.get_truth(padded_base_inputs)

        # Forward pass with prefix || base audios and get hidden states
        adv_hidden_states = self.get_hidden_states(concat_input_values)

        # Compute hidden loss
        hidden_loss = self.loss(adv_hidden_states, self.true_hidden_states)

        return self.finish_attack_step(hidden_loss, prefix_inputs, neg)
