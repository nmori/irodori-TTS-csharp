import os, sys, math, torch, soundfile as sf, numpy as np
sys.path.insert(0, os.environ["IRODORI_TTS_REPO"])
from safetensors.torch import save_file
from irodori_tts.codec import DACVAECodec
from irodori_tts.inference_runtime import _load_audio
weights, wav_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
# synthetic 2.3 s test signal at 44.1 kHz stereo: chirp + harmonics + noise, as 16-bit PCM
sr = 44100; n = int(2.3 * sr); t = np.arange(n) / sr
rng = np.random.default_rng(0)
sig = 0.3*np.sin(2*np.pi*(120*t + 400*t*t)) + 0.1*np.sin(2*np.pi*660*t)*np.sin(2*np.pi*3*t) + 0.02*rng.standard_normal(n)
stereo = np.stack([sig, 0.8*sig + 0.01*rng.standard_normal(n)], 1)
sf.write(wav_path, stereo, sr, subtype="PCM_16")
codec = DACVAECodec.load(repo_id=weights, device="cpu", dtype=torch.float32)
wav, wsr = _load_audio(wav_path)
lat = codec.encode_waveform(wav.unsqueeze(0), sample_rate=int(wsr), normalize_db=-16.0, ensure_max=True)
lat2 = codec.encode_waveform(wav.unsqueeze(0), sample_rate=int(wsr), normalize_db=None, ensure_max=True)
import torchaudio
res = torchaudio.functional.resample(wav.mean(0, keepdim=True), wsr, 48000)
from audiotools import AudioSignal
s = AudioSignal(res.unsqueeze(0), 48000); loud = s.loudness()
with torch.inference_mode():
    audio = codec.decode_latent(lat)
print("latent", lat.shape, "audio", audio.shape, "loud", loud, "hop", codec.model.hop_length)
save_file({"wav": wav.contiguous(), "resampled": res.contiguous(), "loudness": loud.float(), "latent": lat.contiguous(), "latent_nonorm": lat2.contiguous(), "audio": audio.contiguous()}, out)
