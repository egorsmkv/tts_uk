# Text-to-Speech for Ukrainian

Demo: https://huggingface.co/spaces/Yehor/radtts-uk-vocos-demo

## Install

```shell
uv venv --python 3.10

source .venv/bin/activate

uv sync
```

## Install as Python package

```shell
pip install git+https://github.com/egorsmkv/tts_uk
```

## Google Colabs

- CPU: https://colab.research.google.com/drive/1dsQiVhTaNw5lRfUiCZeECMuEbtEEYqbZ?usp=sharing
- GPU: https://colab.research.google.com/drive/1sdCPnZJRNAf12PhPut4gu6T_o6lYaUdo?usp=sharing

## Example

As the code:

```python
import torchaudio

from radtts_uk.inference import synthesis

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

