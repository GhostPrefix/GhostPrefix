from datasets import load_dataset, Audio
import julius
import numpy as np
import torch
import torchaudio

MAX_PADDING = 2.0

MIN_HF_CUTOFF = 1
MAX_HF_CUTOFF = 1000

MIN_LF_CUTOFF = 4000
MAX_LF_CUTOFF = 8000

MIN_NOISE_SNR = 20
MAX_NOISE_SNR = 25


class RobustPipeline():
    def __init__(self, attack):
        from .base import BaseAttack  # Avoid circular import

        self.attack: BaseAttack = attack

        self.load_ir_dataset()

    def load_ir_dataset(self):
        if not self.attack.params["robust"]:
            return

        torch.backends.cuda.cufft_plan_cache[0].max_size = 0

        self.ir_dataset = load_dataset(self.attack.params["ir_dataset"], split="train")
        self.ir_dataset = self.ir_dataset.cast_column("audio", Audio(sampling_rate=self.attack.params["sr"]))

    def reset(self, batch_size: int):
        if not self.attack.params["robust"]:
            return

        # Select random padding for start and end of audio
        self.start_padding = torch.zeros(batch_size, torch.randint(int(self.attack.params["sr"] * MAX_PADDING), (1,)))
        self.end_padding = torch.zeros(batch_size, torch.randint(int(self.attack.params["sr"] * MAX_PADDING), (1,)))

        # Get random batch of impulse responses
        rand_irs = self.ir_dataset.select(torch.randint(len(self.ir_dataset), (batch_size,)))
        ir_batch = [s["audio"]["array"] for s in rand_irs]

        self.ir_input_values = self.attack.model.preprocess_for_attack(ir_batch)
        self.ir_input_values /= torch.linalg.vector_norm(self.ir_input_values, ord=2, dim=-1, keepdim=True)
        self.ir_input_values = self.ir_input_values.float()

        # Set parameters for band-pass filtering
        min_freqs = torch.randint(MIN_HF_CUTOFF, MAX_HF_CUTOFF + 1, (batch_size,))
        max_freqs = torch.randint(MIN_LF_CUTOFF, MAX_LF_CUTOFF + 1, (batch_size,))

        self.select_idx = torch.arange(batch_size)
        self.high_pass = julius.HighPassFilters([f / self.attack.params["sr"] for f in min_freqs])
        self.low_pass = julius.LowPassFilters([f / self.attack.params["sr"] for f in max_freqs])

        # Set parameters for Gaussian noise addition
        self.noise_snrs = torch.randint(MIN_NOISE_SNR, MAX_NOISE_SNR + 1, (batch_size,))

    def apply(self, input_values: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if not self.attack.params["robust"]:
            return input_values

        return_numpy = False
        if isinstance(input_values, np.ndarray):
            input_values = torch.from_numpy(input_values)
            return_numpy = True

        return_squeezed = False
        if len(input_values.shape) == 1:
            input_values = input_values.unsqueeze(0)
            return_squeezed = True

        d = input_values.device
        t = input_values.dtype

        input_values = input_values.float()

        # Apply padding
        input_values = torch.concatenate((self.start_padding.to(d), input_values, self.end_padding.to(d)), 1)

        # Apply impulse responses
        input_values = torchaudio.functional.fftconvolve(input_values, self.ir_input_values.to(d))

        # Apply band-pass filtering
        input_values = self.high_pass.to(d)(input_values)[self.select_idx, self.select_idx, :]
        input_values = self.low_pass.to(d)(input_values)[self.select_idx, self.select_idx, :]

        # Apply Gaussion noise
        noise = torch.randn_like(input_values)
        input_values = torchaudio.functional.add_noise(input_values, noise=noise, snr=self.noise_snrs.to(d))

        input_values = input_values.clamp(-1.0, 1.0)
        input_values = input_values.to(t)

        if return_squeezed:
            input_values = input_values.squeeze()
        if return_numpy:
            input_values = self.attack.inputs_to_audio(input_values)
        return input_values

    def apply_and_cat(self, prefix_inputs: torch.Tensor | np.ndarray, base_inputs: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        trans_prefix_inputs = self.apply(prefix_inputs)

        if isinstance(trans_prefix_inputs, torch.Tensor) and isinstance(base_inputs, torch.Tensor):
            return torch.concatenate((trans_prefix_inputs, base_inputs), 1)
        else:
            return np.concatenate((trans_prefix_inputs, base_inputs))
