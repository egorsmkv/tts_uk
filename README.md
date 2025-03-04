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

High-fidelity speech synthesis for Ukrainian using modern neural networks.

## Demo

Check out our demo on [Hugging Face space](https://huggingface.co/spaces/Yehor/radtts-uk-vocos-demo) or just [listen to samples here](https://huggingface.co/spaces/speech-uk/listen-tts-voices).

## Notes

- Multispeaker: 2 female + 1 male voices;
- Tested on **Windows** and **WSL**.

## Installation

```shell
# from PyPI
pip install tts-uk

# from GitHub
pip install git+https://github.com/egorsmkv/tts_uk

# using GitHub
git clone https://github.com/egorsmkv/tts_uk
cd tts_uk
uv sync
```

Read [uv's installation](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) section.

Also, you can [download the repository](https://github.com/egorsmkv/tts_uk/archive/refs/heads/main.zip) as a ZIP archive.

## Getting started

Code example:

```python
import torchaudio

from tts_uk.inference import synthesis

mels, vocos_wav_gen, stats = synthesis(
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

torchaudio.save("audio.wav", vocos_wav_gen.cpu(), 44_100, encoding="PCM_S")
```

Use these Google colabs:

- [CPU inference](https://colab.research.google.com/drive/1dsQiVhTaNw5lRfUiCZeECMuEbtEEYqbZ?usp=sharing)
- [GPU inference](https://colab.research.google.com/drive/1sdCPnZJRNAf12PhPut4gu6T_o6lYaUdo?usp=sharing)

Or run synthesis in a terminal:

```shell
uv run example.py
```

## Get help and support

Please feel free to connect with us using [the Issues section](https://github.com/egorsmkv/tts_uk/issues).

## License

Code has the MIT license.

## Authors

- Acoustic model: [Yehor Smoliakov](https://github.com/egorsmkv)
- Vocoder: [Serhiy Stetskovych](https://github.com/patriotyk)

## Community

- **Discord**: https://bit.ly/discord-uds
- Speech Recognition: https://t.me/speech_recognition_uk
- Speech Synthesis: https://t.me/speech_synthesis_uk
