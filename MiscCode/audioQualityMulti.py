#Check audio quality of multiple files

import os
import numpy as np
import pandas as pd
import torchaudio
import torch
import librosa
from torchmetrics.audio import SignalNoiseRatio

def calculate_snr(data):
    signal_power = torch.mean(data**2)
    noise_power = torch.var(data)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))  # avoid division by zero
    return snr.item()

def resample_audio(data, orig_sr, target_sr=16000):
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(data)

def calculate_spectral(data):
    spectrogram_transform = torchaudio.transforms.Spectrogram()
    spectrogram = spectrogram_transform(data)
    power_spectrum = spectrogram.pow(2.0)
    geometric_mean = torch.exp(torch.mean(torch.log(power_spectrum + 1e-10), dim=-1))  # geometric mean
    arithmetic_mean = torch.mean(power_spectrum, dim=-1)  # arithmetic mean
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)  # avoid division by zero
    return spectral_flatness.mean().item()

def process_audio_files(audio_path, output_csv):
    results = []
    torch_snr = SignalNoiseRatio()

    for filename in os.listdir(audio_path):
        if (filename.endswith(".WAV") or filename.endswith(".wav")) and not filename.startswith('._'):
            print(f'Processing: {filename}')
            file_path = os.path.join(audio_path, filename)
            waveform, sr = torchaudio.load(file_path)

            if sr != 16000:
                data_resampled = resample_audio(waveform, sr, 16000)
                # snr_original = calculate_snr(waveform)
                # snr_resampled = calculate_snr(data_resampled)
                snr_original = torch_snr(waveform, waveform - torch.mean(waveform)).item()
                snr_resampled = torch_snr(data_resampled, data_resampled - torch.mean(data_resampled)).item()
                spc = calculate_spectral(waveform)
                spc_re = calculate_spectral(data_resampled)
                results.append({'filename': filename, 'SNR_original': snr_original, 'SNR_resampled': snr_resampled, 'SF_original': spc, 'SF_resampled': spc_re})
            else:
                # snr = calculate_snr(waveform)
                snr = torch_snr(waveform, waveform - torch.mean(waveform)).item()
                spc = calculate_spectral(waveform)
                results.append({'filename': filename+'not_resampled', 'SNR_original': snr, 'SNR_resampled': snr, 'SF_original': spc, 'SF_resampled': spc})

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f'SNR values saved to {output_csv}')

audio_path = '~/WavFiles_extraAudioAnalysis'
output_path = os.makedirs(f'{audio_path}/audioQuality', exist_ok=True)
output_csv = f'{audio_path}/audioQuality/snr_values.csv'

process_audio_files(audio_path, output_csv)