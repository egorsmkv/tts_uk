<p align="center">
  <a href="https://github.com/egorsmkv/tts_uk">
    <img loading="lazy" alt="tts_uk" src="https://github.com/egorsmkv/tts_uk/raw/main/assets/logo_github.jpg" width="100%"/>
  </a>
</p>

# Text-to-Speech for Ukrainian

[![CI](https://github.com/egorsmkv/tts_uk/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/egorsmkv/tts_uk/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/tts_uk)](https://pypi.org/project/tts_uk/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tts_uk)](https://pypi.org/project/tts_uk/)
[![License MIT](https://img.shields.io/github/license/egorsmkv/tts_uk)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/tts_uk/month)](https://pepy.tech/projects/tts_uk)
[![speech-uk](https://img.shields.io/badge/speech-uk-blue)](https://huggingface.co/speech-uk)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/Yehor/radtts-uk-vocos-demo)

High-fidelity speech synthesis for Ukrainian using modern neural networks.

## Demo

Check out our demo on [Hugging Face space](https://huggingface.co/spaces/Yehor/radtts-uk-vocos-demo) or just [listen to samples here](https://huggingface.co/spaces/speech-uk/listen-tts-voices).

## Features

- Multi-speaker model: 2 **female** (Tetiana, Lada) + 1 **male** (Mykyta) voices;
- Fine-grained control over speech parameters, including duration, fundamental frequency (F0), and energy;
- High-fidelity speech generation using the [RAD-TTS++](https://github.com/egorsmkv/radtts-uk) acoustic model;
- Fast vocoding using [Vocos](https://github.com/gemelo-ai/vocos);
- Synthesizes long sentences effectively;
- Supports a sampling rate of 44.1 kHz;
- Tested on Linux environments and **Windows**/**WSL**;
- Python API (requires Python 3.9 or later);
- CUDA-enabled for GPU acceleration.

## Installation

```shell
# Install from PyPI
pip install tts-uk

# OR, for the latest development version:
pip install git+https://github.com/egorsmkv/tts_uk

# OR, use git and local setup
git clone https://github.com/egorsmkv/tts_uk
cd tts_uk
uv sync # uv will handle the virtual environment
```

Read [uv's installation](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) section.

Also, you can [download the repository](https://github.com/egorsmkv/tts_uk/archive/refs/heads/main.zip) as a ZIP archive.

## Getting started

Code example:

```python
import torchaudio

from tts_uk.inference import synthesis

sampling_rate = 44_100

# Perform the synthesis, `synthesis` function returns:
# - mels: Mel spectrograms of the generated audio.
# - wave: The synthesized waveform by a Vocoder as a PyTorch tensor.
# - stats: A dictionary containing synthesis statistics (processing time, duration, speech rate, etc).
mels, wave, stats = synthesis(
    text="Ви можете протестувати синтез мовлення українською мовою. Просто введіть текст, який ви хочете прослухати.",
    voice="tetiana",  # tetiana, mykyta, lada
    n_takes=1,
    use_latest_take=False,
    token_dur_scaling=1,
    f0_mean=0,
    f0_std=0,
    energy_mean=0,
    energy_std=0,
    sigma_decoder=0.8,
    sigma_token_duration=0.666,
    sigma_f0=1,
    sigma_energy=1,
)

print(stats)

# Save the generated audio to a WAV file.
torchaudio.save("audio.wav", wave.cpu(), sampling_rate, encoding="PCM_S")
```

Use these Google colabs:

- [CPU inference](https://colab.research.google.com/drive/1dsQiVhTaNw5lRfUiCZeECMuEbtEEYqbZ?usp=sharing)
- [GPU inference](https://colab.research.google.com/drive/1sdCPnZJRNAf12PhPut4gu6T_o6lYaUdo?usp=sharing) on T4 card

Or run synthesis in a terminal:

```shell
uv run example.py
```

If you need to synthesize articles we recommend consider [wtpsplit](https://github.com/segment-any-text/wtpsplit).

## Get help and support

Please feel free to connect with us using [the Issues section](https://github.com/egorsmkv/tts_uk/issues).

## License

Code has the MIT license.

## Model authors

### Acoustic

- [Yehor Smoliakov](https://github.com/egorsmkv), [HF profile](https://huggingface.co/Yehor) 

### Vocoder

- [Serhiy Stetskovych](https://github.com/patriotyk), [HF profile](https://huggingface.co/patriotyk) 

## Community

- Discord: https://bit.ly/discord-uds
- Speech Recognition: https://t.me/speech_recognition_uk
- Speech Synthesis: https://t.me/speech_synthesis_uk

Also, follow [our Speech-UK initiative](https://huggingface.co/speech-uk) on Hugging Face!

## Acknowledgements

- [RAD-TTS by NVIDIA](https://github.com/NVIDIA/radtts)
- [Vocos fork by @langtech-bsc](https://github.com/langtech-bsc/vocos/tree/matcha)
