# GhostPrefix

Source code for the paper, "GhostPrefix: Synchronization-Free Universal Prefix Attacks for ASR Manipulation"

by *Anonymous Authors*, in submission to [Network and Distributed System Security Symposium (NDSS) 2027](https://www.ndss-symposium.org/ndss2027/).

Visit our [website](https://ghostprefix.github.io/) for artifacts and demos.

In this repository, we provide the complete code for training and testing **GhostPrefix** adversarial audio prefixes. We also provide our own optimized prefixes in `prefixes/`.

## Installation

Clone this repository and install dependencies with `pip install -r requirements.txt`.

For GPU support, you may need to install PyTorch separately following the instructions [here](https://pytorch.org/get-started/locally/).

If you run into issues installing `omnilingual-asr` and `fairseq2` may also need to install `fairseq2` separately following the instructions [here](https://github.com/facebookresearch/fairseq2).

## Usage

To generate your own **GhostPrefix** adversarial audio prefixes, first edit one of the .yaml files in `configs`. There are three of them, one for each attack mode: `prompted.yaml`, `targeted.yaml`, and `untargeted.yaml`.

1. Set `root` and `cache` to your root and cache directories, respectively.
2. Set `model_path` to the ASR model you want to attack. We attacked the following 13 models in our paper:
    - facebook/wav2vec2-base-960h
    - facebook/wav2vec2-large-960h
    - facebook/wav2vec2-large-960h-lv60-self
    - facebook/omniASR_CTC_300M_v2
    - facebook/omniASR_CTC_1B_v2
    - facebook/omniASR_LLM_Unlimited_300M_v2
    - facebook/omniASR_LLM_Unlimited_1B_v2
    - openai/whisper-tiny
    - openai/whisper-base
    - openai/whisper-small
    - openai/whisper-medium
    - openai/whisper-large-v3
    - openai/whisper-large-v3-turbo
3. If you wish to tailor the prefix to an environmental sound template, set `initial_prefix` to the path to a .wav file. It could be a car horn sound effect, notification tone, or any other environmental sound. **GhostPrefix** will initialize the prefix to that sound template and perform a Carlini–Wagner-style L2 attack so that it remains close to that sound.
4. To train a prefix that's robust to over-the-air playback, set `robust` to True and set `ir_dataset` to the path to a directory of room impulse response (RIR) files. We use the MIT McDermott dataset, available [here](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html).

**GhostPrefix** currently only supports two Hugging Face datasets, `librispeech_asr` and `facebook/multilingual_librispeech`. To use the latter, `name` can be set to one of the languages included in MLS (`german`, `dutch`, etc.). To use a different dataset, you may have to edit the code to support it.

Additionally, while we use MSE loss on hidden states for our untargeted attack mode, we also implement a traditional untargeted attack that uses a token-level loss. To use it, set `attack` to `!!python/name:attacks.TraditionalUntargetedAttack` in `configs/untargeted.yaml`.

---

Use `train.py` with one of the config files to begin training the prefix, for example:

```bash
python train.py configs/prompted.yaml
```

Prefix checkpoints, saved optimizer states, and the training log will be saved in the directory specified by `save_folder` in the config file. After each epoch, evaluation results will be saved in `eval_folder`.

---

To evaluate a prefix on the test dataset, use `evaluate.py`:

```bash
python evaluate.py configs/prompted.yaml path/to/prefix.wav optional/path/to/output.log
```

By default, this will also evaulate the performance of the ASR model with no attack. If that is too time-consuming, add `eval_no_atk=False` to the `attack.evaluate()` function call in line 28 of `evaluate.py`.

---

To simply perform ASR on an audio file, use `asr.py`:

```bash
python asr.py configs/prompted.yaml path/to/audio.wav
```

This won't perform any attack. Just make sure `model_path` in the config file is set to the ASR model you want to use.

---

👻
