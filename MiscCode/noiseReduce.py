#Test script for noisereduction
import numpy as np
import noisereduce as nr
import soundfile as sf
import librosa

audio, sample_rate = sf.read('~/extra/example.wav')

print(f'Audio shape: {audio.shape}')
print(f'Sample rate: {sample_rate}')

#convert to mono
audio_mono = np.mean(audio, axis=1)

#resample to 16khz
audio_16k = librosa.resample(audio_mono, orig_sr=sample_rate, target_sr=16000)

#process in chunks
def reduce_noise_in_chunks(audio, sample_rate, chunk_size=5):
    chunk_length = int(sample_rate * chunk_size)
    processed_audio = []
    
    for start in range(0, len(audio), chunk_length):
        end = start + chunk_length
        chunk = audio[start:end]
        reduced_chunk = nr.reduce_noise(
            y=chunk,
            sr=sample_rate,
            prop_decrease=0.8,
            time_constant_s=1.0,
            freq_mask_smooth_hz=500
        )
        processed_audio.append(reduced_chunk)
    
    return np.concatenate(processed_audio)

reduced_noise_audio = reduce_noise_in_chunks(audio_16k, 16000)

sf.write('~/extra/example_reduced.wav', reduced_noise_audio, 16000)