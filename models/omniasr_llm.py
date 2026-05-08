from fairseq2.datasets.batch import Seq2SeqBatch
import torch

from .omniasr_base import OmniAsrBaseAttack, OmniAsrBaseAttackOnHiddenStates


class OmniAsrLlmAttack(OmniAsrBaseAttack):
    def __init__(self, params, robust):
        super().__init__(params, robust)

        self.lang = "eng_Latn"

    def infer_batch(self, input_values: torch.Tensor) -> list[str]:
        with torch.no_grad():
            input_values = self.normalize_audio_inputs(input_values)

            batch_layout = self.get_batch_layout(input_values)

            batch = Seq2SeqBatch(
                source_seqs=input_values,
                source_seq_lens=batch_layout.seq_lens,
                target_seqs=torch.zeros((input_values.shape[0], 1), dtype=torch.int64, device=self.device),
                target_seq_lens=[1] * input_values.shape[0],
                example={"lang": [self.lang] * input_values.shape[0]},
            )

            transcriptions = self.pipeline._apply_model_wav2vec2llama(batch)
            return transcriptions

    def attack_step(self, prefix_inputs: torch.Tensor, base_inputs: torch.Tensor, neg: bool = False) -> torch.Tensor:
        concat_input_values = self.prepend_to_base_inputs(prefix_inputs, base_inputs)
        concat_input_values = self.normalize_audio_inputs(concat_input_values)

        # Create batch for OmniASR
        batch_layout = self.get_batch_layout(concat_input_values)
        batch = Seq2SeqBatch(
            source_seqs=concat_input_values,
            source_seq_lens=batch_layout.seq_lens,
            target_seqs=self.target_ids,
            target_seq_lens=self.target_layout.seq_lens,
            example={"lang": [self.lang] * concat_input_values.shape[0]}
        )

        self.pipeline.streaming_config.is_streaming = False

        # Forward pass with prefix || base audios
        loss = self.pipeline.model(batch)

        self.pipeline.streaming_config.is_streaming = True

        return self.finish_attack_step(loss, prefix_inputs, neg)


class OmniAsrLlmAttackOnHiddenStates(OmniAsrBaseAttackOnHiddenStates, OmniAsrLlmAttack):

    def get_hidden_states(self, audio_inputs: torch.Tensor) -> torch.Tensor:
        audio_inputs = self.normalize_audio_inputs(audio_inputs)

        batch_layout = self.get_batch_layout(audio_inputs)

        seqs, seqs_layout, _ = self.pipeline.model.encoder_frontend.extract_features(audio_inputs, batch_layout)
        seqs, _ = self.pipeline.model.encoder_frontend.process_features(seqs, seqs_layout)
        hidden_states = self.pipeline.model.encoder(seqs, seqs_layout)

        return hidden_states
