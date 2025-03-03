import os
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

# Vocoder
from vocos.pretrained import Vocos

from .common import update_params
from .config import config
from .data import TextProcessor

# RAD-TTS code
from .radtts import RADTTS
from .torch_env import device


def download_file_from_repo(
    repo_id: str,
    filename: str,
    local_dir: str = ".",
    repo_type: str = "model",
) -> str:
    try:
        os.makedirs(local_dir, exist_ok=True)

        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            cache_dir=None,
            force_download=False,
            repo_type=repo_type,
        )

        return file_path
    except Exception as e:
        raise Exception(f"An error occurred during download: {e}") from e


download_file_from_repo(
    "Yehor/radtts-uk",
    "radtts-pp-dap-model/model_dap_84000_state.pt",
    "./models/",
)

# Init the model
params = []

# Load the config
update_params(config, params)

data_config = config["data_config"]
model_config = config["model_config"]

# Load vocoder
vocos_config = hf_hub_download(
    "patriotyk/vocos-mel-hifigan-compat-44100khz", "config.yaml"
)
vocos_model = hf_hub_download(
    "patriotyk/vocos-mel-hifigan-compat-44100khz", "pytorch_model.bin"
)

vocos_model_path = Path(vocos_model)
state_dict = torch.load(vocos_model_path, weights_only=True, map_location="cpu")

vocos = Vocos.from_hparams(vocos_config).to(device)
vocos.load_state_dict(state_dict, strict=True)
vocos.eval()

# Load RAD-TTS
radtts = RADTTS(**model_config).to(device)
radtts.enable_inverse_cache()  # cache inverse matrix for 1x1 invertible convs

radtts_model_path = Path("models/radtts-pp-dap-model/model_dap_84000_state.pt")

checkpoint_dict = torch.load(radtts_model_path, weights_only=True, map_location="cpu")
state_dict = checkpoint_dict["state_dict"]

radtts.load_state_dict(state_dict, strict=False)
radtts.eval()

radtts_params = f"{sum(param.numel() for param in radtts.parameters()):,}"
vocos_params = f"{sum(param.numel() for param in vocos.parameters()):,}"

print(f"Loaded checkpoint (RAD-TTS++), number of parameters: {radtts_params}")
print(f"Loaded checkpoint (Vocos), number of parameters: {vocos_params}")

text_processor = TextProcessor(
    data_config["training_files"],
    **dict(
        (k, v)
        for k, v in data_config.items()
        if k not in ["training_files", "validation_files"]
    ),
)

voices = {
    "lada": 0,
    "mykyta": 1,
    "tetiana": 2,
}


def synthesis(
    text,
    voice,
    n_takes,
    use_latest_take,
    token_dur_scaling,
    f0_mean,
    f0_std,
    energy_mean,
    energy_std,
    sigma_decoder,
    sigma_token_duration,
    sigma_f0,
    sigma_energy,
):
    if not text:
        raise ValueError("Please paste your text.")

    speaker = speaker_text = speaker_attributes = voice.lower()

    tensor_text = torch.LongTensor(text_processor.tp.encode_text(text)).to(device)
    speaker_tensor = torch.LongTensor([voices[speaker]]).to(device)

    speaker_id = speaker_id_text = speaker_id_attributes = speaker_tensor

    if speaker_text is not None:
        speaker_id_text = torch.LongTensor([voices[speaker_text]]).to(device)

    if speaker_attributes is not None:
        speaker_id_attributes = torch.LongTensor([voices[speaker_attributes]]).to(
            device
        )

    inference_start = time.time()

    mels = []
    for n_take in range(n_takes):
        print(f"Inferencing take {n_take + 1}")

        with torch.autocast(device, enabled=False):
            with torch.inference_mode():
                outputs = radtts.infer(
                    speaker_id,
                    tensor_text[None],
                    sigma_decoder,
                    sigma_token_duration,
                    sigma_f0,
                    sigma_energy,
                    token_dur_scaling,
                    token_duration_max=100,
                    speaker_id_text=speaker_id_text,
                    speaker_id_attributes=speaker_id_attributes,
                    f0_mean=f0_mean,
                    f0_std=f0_std,
                    energy_mean=energy_mean,
                    energy_std=energy_std,
                )

                mels.append(outputs["mel"])

    wav_gen_all = []
    for mel in mels:
        wav_gen_all.append(vocos.decode(mel))

    if use_latest_take:
        wav_gen = wav_gen_all[-1]  # Get the latest generated wav
    else:
        wav_gen = torch.cat(wav_gen_all, dim=1)  # Concatenate all the generated wavs

    duration = len(wav_gen[0]) / 44_100

    elapsed_time = time.time() - inference_start
    rtf = elapsed_time / duration

    speed_ratio = duration / elapsed_time
    speech_rate = len(text.split(" ")) / duration

    stats = {
        "rtf": rtf,
        "time": elapsed_time,
        "audio_duration": duration,
        "speed_ratio": speed_ratio,
        "speech_rate": speech_rate,
    }

    return [mels, wav_gen.cpu(), stats]
