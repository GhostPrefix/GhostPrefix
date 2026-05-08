import numpy as np
import torch
from torch import nn

from attacks.robust import RobustPipeline


class BaseModelAttack():
    def __init__(self, params, robust: RobustPipeline):
        self.params = params
        self.robust = robust

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.initial_prefix_inputs: torch.Tensor | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def prepend_to_base_inputs(self, prefix_inputs: torch.Tensor, base_inputs: torch.Tensor) -> torch.Tensor:
        repeat_prefix_inputs = prefix_inputs.repeat(base_inputs.shape[0], 1)
        concat_inputs = self.robust.apply_and_cat(repeat_prefix_inputs, base_inputs)  # Apply transformations if applicable
        return concat_inputs

    def normalize_audio_inputs(self, input_values: torch.Tensor) -> torch.Tensor:
        # Normalize audio (zero mean, unit variance)
        means = input_values.mean(1).view(-1, 1)
        variances = input_values.var(1).view(-1, 1)

        norm_input_values = (input_values - means) / torch.sqrt(variances + 1e-7)
        return norm_input_values

    def transform_text(self, text: str) -> str:
        pass

    def get_targets(self, transcriptions: list[str]):
        pass

    def preprocess_for_attack(self, audio: np.ndarray | list | tuple) -> torch.Tensor:
        pass

    def infer_batch(self, input_values: torch.Tensor) -> list[str]:
        pass

    def infer_batch_with_prefix(self, prefix_input_values: torch.Tensor, base_input_values: torch.Tensor) -> list[str]:
        pass

    def infer_single(self, audio_input: np.ndarray) -> str:
        pass

    def attack_step(self, prefix_inputs: torch.Tensor, base_inputs: torch.Tensor, neg: bool = False) -> torch.Tensor:
        pass

    def finish_attack_step(self, loss: torch.Tensor, prefix_inputs: torch.Tensor, neg: bool) -> torch.Tensor:
        # Negate loss if needed
        if neg:
            loss = -loss

        # If applicable, perform Carlini-Wagner attack with L2 loss
        if self.params["initial_prefix"]:
            loss += self.params["c"] * torch.linalg.vector_norm(prefix_inputs - self.initial_prefix_inputs, ord=2)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Ensure perturbation stays within allowed range
        with torch.no_grad():
            if self.params["initial_prefix"]:
                delta = torch.clamp(prefix_inputs - self.initial_prefix_inputs, -self.params["epsilon"], self.params["epsilon"])
                prefix_inputs.copy_(torch.clamp(self.initial_prefix_inputs + delta, -1.0, 1.0))
            else:
                prefix_inputs.clamp_(-self.params["epsilon"], self.params["epsilon"])

        return loss


class BaseModelAttackOnHiddenStates(BaseModelAttack):
    def __init__(self, params, robust):
        super().__init__(params, robust)

        self.loss = nn.MSELoss()

        self.true_hidden_states: torch.Tensor | None = None

    def get_truth(self, original_inputs: torch.Tensor):
        pass
