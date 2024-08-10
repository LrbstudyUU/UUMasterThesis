import torchaudio
import torch
import scipy.io.wavfile as wav
import soundfile as sf

import soundfile as sf
import numpy as np
import logmmse
import torch
from torchmetrics.audio import SignalNoiseRatio

# Function to convert stereo to mono
def stereo_to_mono(audio):
    if audio.ndim > 1:
        return np.mean(audio, axis=1)
    return audio

# Function to load and enhance audio
def load_and_enhance_audio(file_path):
    # Load the audio file
    audio, sample_rate = sf.read(file_path)
    
    # Convert to mono if necessary
    mono_audio = stereo_to_mono(audio)
    
    # Ensure the audio is in the correct format (float32)
    mono_audio = mono_audio.astype(np.float32)
    
    # Apply logMMSE noise reduction
    enhanced_audio = logmmse.logmmse(mono_audio, sample_rate)
    
    return mono_audio, enhanced_audio, sample_rate

# Function to trim audio to the same length
def trim_to_same_length(audio1, audio2):
    min_length = min(len(audio1), len(audio2))
    return audio1[:min_length], audio2[:min_length]

# Function to calculate SNR
def calculate_snr(clean_audio, noisy_audio):
    snr_metric = SignalNoiseRatio()
    
    # Convert audio to torch tensors
    clean_audio_tensor = torch.tensor(clean_audio, dtype=torch.float32)
    noisy_audio_tensor = torch.tensor(noisy_audio, dtype=torch.float32)
    
    # Calculate SNR
    snr_value = snr_metric(clean_audio_tensor, noisy_audio_tensor)
    
    return snr_value.item()

# Load and enhance the audio file
file_path = '~/WavFiles_extraAudioAnalysis/session.wav'
original_audio, enhanced_audio, sample_rate = load_and_enhance_audio(file_path)

# Trim the audio signals to the same length
original_audio, enhanced_audio = trim_to_same_length(original_audio, enhanced_audio)

# Calculate SNR before and after enhancement
snr_before = calculate_snr(original_audio, original_audio)
snr_after = calculate_snr(original_audio, enhanced_audio)

print(f"SNR before enhancement: {snr_before:.2f} dB")
print(f"SNR after enhancement: {snr_after:.2f} dB")

# def calculate_spectral_flatness_verification(data):
#     spectrogram_transform = torchaudio.transforms.Spectrogram()
#     spectrogram = spectrogram_transform(data)
#     power_spectrum = spectrogram.pow(2.0)
#     geometric_mean = torch.exp(torch.mean(torch.log(power_spectrum + 1e-10), dim=-1))  # geometric mean
#     arithmetic_mean = torch.mean(power_spectrum, dim=-1)  # arithmetic mean
#     spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)  # avoid division by zero
#     return spectral_flatness.mean().item()

# # Example usage
# waveform, orig_sr = torchaudio.load('~/WavFiles_extraAudioAnalysis/4013-1-20-6-2017.wav')
# resampled_waveform = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=16000)(waveform)
# sf_original = calculate_spectral_flatness_verification(waveform)
# sf_resampled = calculate_spectral_flatness_verification(resampled_waveform)

# print(f'Original SF: {sf_original}, Resampled SF: {sf_resampled}')


# snr = SignalNoiseRatio()

# audio, sr = sf.read('~/WavFiles_extraAudioAnalysis/session.wav')

# #audio, sr = torchaudio.load('~/WavFiles_extraAudioAnalysis/sessionwav')
# #sample_rate, audio = wav.read('~/WavFiles_extraAudioAnalysis/session.wav')
# print(snr(audio, audio - torch.mean(audio)).item())

#enhanced_audio = logmmse.logmmse(audio, sr)
#en_audio, en_sr = torchaudio.load(enhanced_audio)
#print(snr(enhanced_audio, enhanced_audio - torch.mean(audio)).item())



