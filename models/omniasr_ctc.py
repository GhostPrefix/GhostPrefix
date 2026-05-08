import torch

from .omniasr_base import OmniAsrBaseAttack, OmniAsrBaseAttackOnHiddenStates


class OmniAsrCtcAttack(OmniAsrBaseAttack):

    def infer_batch(self, input_values: torch.Tensor) -> list[str]:
        with torch.no_grad():
            input_values = self.normalize_audio_inputs(input_values)

            batch_layout = self.get_batch_layout(input_values)

            logits, bl_out = self.pipeline.model(input_values, batch_layout)

        pred_ids = torch.argmax(logits, dim=-1)
        transcriptions = []

        for i in range(pred_ids.shape[0]):
            # Create a mask for where consecutive elements differ (CTC decoding)
            # First element is always True, then compare with previous elements
            seq = pred_ids[i][: bl_out.seq_lens[i]]
            mask = torch.ones(seq.shape[0], dtype=torch.bool, device=seq.device)
            mask[1:] = seq[1:] != seq[:-1]

            # Use the mask to select non-duplicate tokens
            decoded_ids = seq[mask]
            transcriptions.append(self.pipeline.token_decoder(decoded_ids))

        return transcriptions

    def attack_step(self, prefix_inputs: torch.Tensor, base_inputs: torch.Tensor, neg: bool = False) -> torch.Tensor:
        concat_input_values = self.prepend_to_base_inputs(prefix_inputs, base_inputs)
        concat_input_values = self.normalize_audio_inputs(concat_input_values)

        # Forward pass with prefix || base audios
        batch_layout = self.get_batch_layout(concat_input_values)
        loss = self.pipeline.model(concat_input_values, batch_layout, self.target_ids, self.target_layout)

        return self.finish_attack_step(loss, prefix_inputs, neg)


class OmniAsrCtcAttackOnHiddenStates(OmniAsrBaseAttackOnHiddenStates, OmniAsrCtcAttack):

    def get_hidden_states(self, audio_inputs: torch.Tensor) -> torch.Tensor:
        audio_inputs = self.normalize_audio_inputs(audio_inputs)

        batch_layout = self.get_batch_layout(audio_inputs)

        seqs, seqs_layout, _ = self.pipeline.model.encoder_frontend.extract_features(audio_inputs, batch_layout)
        seqs, _ = self.pipeline.model.encoder_frontend.process_features(seqs, seqs_layout)
        hidden_states = self.pipeline.model.encoder(seqs, seqs_layout)

        return hidden_states
