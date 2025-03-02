# RAD-TTS UK library

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

```python
import torchaudio

from radtts_uk.inference import synthesis

wav_gen, stats = synthesis(
    "Ви можете протестувати синтез мовлення українською мовою. Просто введіть текст, який ви хочете прослухати.",  # text
    "Tetiana",  # voice
    1,  # n_takes
    False,  # use_latest_take
    1,  # token_dur_scaling
    0,  # f0_mean
    0,  # f0_std
    0,  # energy_mean
    0,  # energy_std
    0.8,  # sigma_decoder
    0.666,  # sigma_token_duration
    1,  # sigma_f0
    1,  # sigma_energy
)

print(stats)

torchaudio.save("audio.wav", wav_gen.cpu(), 44_100, encoding="PCM_S")
```
