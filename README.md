# Text-to-Speech for Ukrainian

Check out our demo on [Hugging Face space](https://huggingface.co/spaces/Yehor/radtts-uk-vocos-demo).

## Notes

- Multispeaker: 2 female + 1 male voices;
- Tested on **Windows** and **WSL**.

## Install

```shell
uv sync
```

Read [uv's installation](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) section.

## Install as Python package

```shell
# from PyPI
pip install tts-uk

# or from GitHub
pip install git+https://github.com/egorsmkv/tts_uk
```

Or [download the repository](https://github.com/egorsmkv/tts_uk/archive/refs/heads/main.zip) as a ZIP archive.

## Google Colabs

- [CPU inference](https://colab.research.google.com/drive/1dsQiVhTaNw5lRfUiCZeECMuEbtEEYqbZ?usp=sharing)
- [GPU inference](https://colab.research.google.com/drive/1sdCPnZJRNAf12PhPut4gu6T_o6lYaUdo?usp=sharing)

## Example

As the code:

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

Or using a terminal:

```shell
uv run example.py
```

