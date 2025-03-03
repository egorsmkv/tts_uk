import torchaudio

from tts_uk.inference import synthesis

mels, vocos_wav_gen, stats = synthesis(
    text="Ви можете протестувати синтез мовлення українською мовою. Просто введіть текст, який ви хочете прослухати.",
    voice="mykyta",  # tetiana, mykyta, lada
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
