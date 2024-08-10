import torchaudio
import torch
from speechbrain.inference.separation import SepformerSeparation as sep
from tqdm import tqdm
import noisereduce as nr

def split_audio(file_path, chunk_size=10):
    waveform, sample_rate = torchaudio.load(file_path)

    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    sample_rate = 16000

    chunk_length = chunk_size * sample_rate
    chunks = torch.split(waveform, chunk_length, dim=1)
    return chunks, sample_rate

def normalize_waveform(waveform):
    peak = torch.max(torch.abs(waveform))
    if peak > 1.0:
        waveform = waveform / peak
    return waveform

model = sep.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement", 
                         savedir='pretrained_models/sepformer-dns4-16k-enhancement', 
                         run_opts={'device': 'cuda'})

audio_chunks, sr = split_audio('~/audio.wav')
enhanced_chunks = []

#process chunks
for i, chunk in enumerate(tqdm(audio_chunks, desc="Processing chunks")):
    try:
        chunk = chunk.squeeze(0).unsqueeze(0)
        est_sources = model.separate_batch(chunk)
        enhanced_chunk = est_sources[:, :, 0].detach().cpu()
        enhanced_chunk = normalize_waveform(enhanced_chunk)
        
        # Apply noise reduction
        enhanced_chunk_np = enhanced_chunk.numpy().flatten()
        reduced_noise_chunk = nr.reduce_noise(y=enhanced_chunk_np, sr=sr)
        reduced_noise_chunk = torch.tensor(reduced_noise_chunk).unsqueeze(0)
        
        enhanced_chunks.append(reduced_noise_chunk)
    except Exception as e:
        print(f"Error processing chunk {i+1}/{len(audio_chunks)}: {e}")

# Concatenate enhanced chunks into a single waveform
enhanced_waveform = torch.cat(enhanced_chunks, dim=1)
enhanced_waveform = normalize_waveform(enhanced_waveform)

torchaudio.save('~/extra/enhancements/enhanced_audio.wav', enhanced_waveform, sr)

# import whisperx
# import csv
# import json

# def dict_to_csv(data, filename):
#     with open(filename, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file, delimiter=";")
#         writer.writerow(['', 'start', 'end', 'text', 'words', 'speaker'])

#         for row_num, segment in enumerate(data['segments'], start=1):
#             if 'speaker' not in segment.keys():
#                 segment['speaker'] = 'NAN'
#             words_json = json.dumps(segment['words'])
#             writer.writerow([row_num, segment['start'], segment['end'], segment['text'], words_json, segment['speaker']])


# device = "cuda" # needs to change for uni computer - GPU should be used
# batch_size = 5 # reduce if low on GPU mem
# compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
# options = {
#         "max_new_tokens": None,
#         "clip_timestamps": None,
#         "hallucination_silence_threshold": None,
#     }

# model = whisperx.load_model("large-v3", device, compute_type=compute_type, language='nl', asr_options=options) # nl for dutch

# audio = whisperx.load_audio('~/extra/enhancements/enhanced_audio.wav')
# #audio = whisperx.load_audio(padded_audio) #Test for Padding
# result = model.transcribe(audio, batch_size=batch_size)

# # align whisper output
# model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
# result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# # diarization
# diarize_model = whisperx.DiarizationPipeline(use_auth_token='', device=device)#, model_name='pyannote/speaker-diarization')
# diarize_segments = diarize_model(audio, num_speakers=2)

# result = whisperx.assign_word_speakers(diarize_segments, result)

# print(result)
# dict_to_csv(result, '~/extra/enhancements/enhanced_audio_transcript.csv')

# import numpy as np
# import librosa
# from speechmos import aecmos, dnsmos, plcmos

# waveform, sr = torchaudio.load('~/audio.wav')

# if waveform.size(0) > 1:
#     waveform = torch.mean(waveform, dim=0, keepdim=True)

# resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
# waveform = resampler(waveform)
# sr = 16000

# device = 'cuda'

# audio_np = waveform.numpy().flatten()

# dns = dnsmos.run(audio_np, sr=sr, verbose=True)

# plc = plcmos.run(audio_np, sr=sr, verbose=True)

# print(f'DNSMOS Result: {dns}, PLCMOS Result: {plc}')

# import os
# import torchaudio
# import traceback
# import pandas as pd
# from speechmos import dnsmos
# from tqdm import tqdm

# from emailMonitor import EmailBot
# ebot = EmailBot()

# def normalize_audio(audio):
#     max_val = max(abs(audio.max()), abs(audio.min()))
#     return audio / max_val if max_val != 0 else audio

# def process_file(filepath):
#     waveform, sr = torchaudio.load(filepath)

#     if waveform.size(0) > 1:
#         waveform = torch.mean(waveform, dim=0, keepdim=True)

#     resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
#     waveform = resampler(waveform)
#     sr = 16000

#     audio = waveform.numpy().flatten()

#     audio = normalize_audio(audio)

#     result = dnsmos.run(audio, sr=16000, verbose=False)

#     return {
#         'filename': os.path.basename(filepath),
#         'OVRL': result['ovrl_mos'],
#         'SIG': result['sig_mos'],
#         'BAK': result['bak_mos'],
#         'P808': result['p808_mos']
#     }

# #file_paths = ['~/WavFiles_extraAudioAnalysis', '~/WavFiles_extraAudioAnalysis_batch2']

# file_path = '~/WavFiles_extraAudioAnalysis_batch2'
# os.makedirs(f'{file_path}/audioQuality', exist_ok=True)
# output_csv = f'{file_path}/audioQuality/audioQReport.csv'

# results = []

# try:
#     os.makedirs(f'{file_path}/audioQuality', exist_ok=True)
#     output_csv = f'{file_path}/audioQuality/audioQReport.csv'
#     for filename in os.listdir(file_path):
#         if (filename.endswith(".WAV") or filename.endswith(".wav")) and not filename.startswith('._'):
#             print(f'Processing: {filename}')
#             filepath = os.path.join(file_path, filename)
#             result = process_file(filepath)
#             results.append(result)

#     df = pd.DataFrame(results)
#     df.to_csv(output_csv, index=False)

#     print(f"Results saved to {output_csv}")

# except Exception:
#     ebot.setEmailContent(traceback.format_exc())
#     ebot.sendEmail()