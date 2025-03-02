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
