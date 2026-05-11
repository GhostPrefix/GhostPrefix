from datasets import Dataset
import numpy as np
import tqdm

import os
import statistics
import time

from .base import BaseAttack


def is_success(wer: float): return wer == 0.0


class TargetedAttack(BaseAttack):

    def train(self, train_dataset: Dataset, eval_dataset: Dataset | None):
        self.init_logger()
        self.logger.info(f"Begin attack on {self.params['model_path']}")

        prefix_input_values = self.init_prefix()

        # Get targets for training
        target_transcriptions = [self.model.transform_text(self.params["target_transcription"])] * self.params["batch_size"]
        self.model.get_targets(target_transcriptions)

        print(f"\nTarget Transcription: '{target_transcriptions[0]}'\n")
        self.logger.info(f"Target Transcription: '{target_transcriptions[0]}'\n\nParameters: {self.params}\n")

        eval_dir = os.path.join(self.params["root"], self.params["eval_folder"], self.params["model_path"].split("/")[-1])

        start_epoch = self.params["prefix_ckpt"] + 1
        for epoch in tqdm.trange(start_epoch, self.params["epochs"] + 1, desc="Training Prefix", unit="epoch"):
            self.logger.info(f"Epoch {epoch}/{self.params['epochs']}")
            train_dataset.set_epoch(epoch)

            n = 1
            for batch in tqdm.tqdm(train_dataset, desc=f"Epoch {epoch}/{self.params['epochs']}", leave=False, initial=n):
                try:
                    start = time.perf_counter()

                    self.robust.reset(self.params["batch_size"])

                    base_input_values = self.model.preprocess_for_attack(batch["audio"])

                    loss = self.model.attack_step(prefix_input_values, base_input_values)

                    end = time.perf_counter()

                    # Print progress
                    if n % self.params["print_freq"] == 0 and os.path.exists(self.save_dir):
                        adv_transcriptions = self.model.infer_batch_with_prefix(prefix_input_values, base_input_values)
                        wer, success = self.get_results(target_transcriptions, adv_transcriptions, is_success, False)

                        self.logger.info(f"Step {n}, Loss: {loss.item():.4f}, WER: {wer:.4f}, Time: {end - start:.4f} s, ({adv_transcriptions[0]})")
                except:
                    pass

                n += 1

            prefix_audio = self.inputs_to_audio(prefix_input_values)
            if eval_dataset:
                os.makedirs(eval_dir, exist_ok=True)
                avg_wer, asr, _, _ = self.evaluate(prefix_audio, eval_dataset, os.path.join(eval_dir, f"prefix-{epoch}.log"), False, False)
                self.logger.info(f"Evaluated prefix, Avg. WER: {avg_wer}, ASR: {asr * 100} %")

            save_file = self.save_prefix(epoch, prefix_audio)
            self.logger.info(f"Saved prefix checkpoint: {save_file}\n")

        self.logger.info("Training Complete")

    def evaluate(self, prefix_audio: np.ndarray, dataset: Dataset, log_file: str | None = None, print_output: bool = True, eval_no_atk: bool = True) -> tuple[float, float, float, float]:
        target_transcription = self.model.transform_text(self.params["target_transcription"])
        if print_output:
            print(f"\nTarget Transcription: '{target_transcription}'\n")

        log_lines = [f"Evaluating Targeted Attack on {self.params['model_path']}:\n",
                     f"Target Transcription: '{target_transcription}'\n\n"
                     f"Parameters: {self.params}\n\n"]

        wers = []
        successes = []
        wers_no_atk = []
        successes_no_atk = []

        start = time.perf_counter()
        for sample in tqdm.tqdm(dataset, desc="Evaluating Prefix", leave=print_output, unit=" samples"):
            try:
                self.robust.reset(1)

                true_transcription = sample["text"]
                base_transcription = self.model.infer_single(sample["audio"]) if eval_no_atk else None

                adv_audio = self.robust.apply_and_cat(prefix_audio, sample["audio"])
                adv_transcription = self.model.infer_single(adv_audio)
            except:
                continue

            wer, success = self.get_results(target_transcription, adv_transcription, is_success, False)

            if eval_no_atk:
                wer_no_atk, success_no_atk = self.get_results(true_transcription, base_transcription, is_success)
                log_lines.append(f"{sample['id']}, WER: {wer}, Success: {success}, WER (no attk): {wer_no_atk}, Success (no attk): {success_no_atk}, Adv. Transcription: [{adv_transcription}], Ground Truth: [{true_transcription}]\n")
            else:
                wer_no_atk, success_no_atk = np.nan, np.nan
                log_lines.append(f"{sample['id']}, WER: {wer}, Success: {success}, Adv. Transcription: [{adv_transcription}], Ground Truth: [{true_transcription}]\n")

            wers.append(wer)
            successes.append(success)
            wers_no_atk.append(wer_no_atk)
            successes_no_atk.append(success_no_atk)

        avg_wer = statistics.mean(wers)
        asr = statistics.mean(successes)
        avg_wer_no_atk = statistics.mean(wers_no_atk)
        sr_no_atk = statistics.mean(successes_no_atk)

        if print_output:
            print(f"\nAverage WER:               {avg_wer:.4f}")
            print(f"Attack Success Rate:       {asr * 100:.4f} %")
            if eval_no_atk:
                print(f"Average WER (no atk):      {avg_wer_no_atk:.4f}")
                print(f"Trx Success Rate (no atk): {sr_no_atk * 100:.4f} %")
            print(f"Elapsed Time:              {time.perf_counter() - start:.4f} s")

        if log_file:
            with open(log_file, "a") as f:
                f.writelines(log_lines)
                f.write(f"\nAverage WER:               {avg_wer}\n"
                        f"Attack Success Rate:       {asr * 100} %\n")
                if eval_no_atk:
                    f.write(f"Average WER (no atk):      {avg_wer_no_atk}\n"
                            f"Trx Success Rate (no atk): {sr_no_atk * 100} %\n")
                f.write(f"Elapsed Time:              {time.perf_counter() - start} s\n")

        return avg_wer, asr, avg_wer_no_atk, sr_no_atk
