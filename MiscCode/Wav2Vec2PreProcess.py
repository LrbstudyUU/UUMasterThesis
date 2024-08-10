import torchaudio
import torch
import os

folder_path = '~/WhisperXTranscriptTest'

for filename in os.listdir(folder_path):
    if (filename.endswith(".WAV") or filename.endswith(".wav")) and not filename.startswith('._'):
        wav_file_path = os.path.join(folder_path, filename)
        waveform, sampling_rate = torchaudio.load(wav_file_path)

        waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sampling_rate != 16000: # Resample to 16000 because wav2vec2 is trained in 16000, do it outside
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
                waveform = resampler(waveform)
                sampling_rate = 16000
        
        base, ext = os.path.splitext(filename)
        new_name = f"{folder_path}/{base}_prepped{ext}"
        #print(new_name)
        torchaudio.save(new_name,waveform,sampling_rate)