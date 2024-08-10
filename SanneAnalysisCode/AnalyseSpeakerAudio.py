#Utterance affect analysis of audio
import traceback
import torchaudio
import gc

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from scipy.io import wavfile
import os
import pandas as pd
from pydub import AudioSegment

class RegressionHead(nn.Module):
    r"""Classification head."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits
    
def process_func_torchaudio(
        waveform: torch.Tensor, 
        sampling_rate: int, 
        embeddings: bool = False,
) -> torch.Tensor:
    """Predict emotions or extract embeddings from raw audio signal loaded with torchaudio."""
    min_length = 0.25 #minimum audio lenght in seconds, less than this results in wav2vec2 errors
    if waveform.size(1) < min_length * 16000:
        print("Skipping short block")
        return [[None, None, None]]
    input_values = processor(waveform.squeeze(), sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values.to(device)
    with torch.no_grad():
        model.eval()
        print("model eval")
        output = model(input_values)[0 if embeddings else 1]
    output_array = output.detach().cpu().numpy()
    return output_array
    
def combine_utterances(df, max_duration=240): #max_duration in seconds, 4 minutes = 240, 5 = 300
    blocks = []
    current_speaker = None
    current_start = None
    current_end = None
    current_text = []
    
    for _, row in df.iterrows():
        utterance_duration = row['end'] - row['start']
        
        if row['speaker'] == current_speaker and current_end + utterance_duration <= current_start + max_duration:
            current_end = row['end']
            current_text.append(row['text'])
        else:
            if current_speaker is not None:
                blocks.append({
                    'start': current_start,
                    'end': current_end,
                    'text': ' '.join(current_text),
                    'speaker': current_speaker
                })
            current_speaker = row['speaker']
            current_start = row['start']
            current_end = row['end']
            current_text = [row['text']]
    
    if current_speaker is not None:
        blocks.append({
            'start': current_start,
            'end': current_end,
            'text': ' '.join(current_text),
            'speaker': current_speaker
        })
    
    return pd.DataFrame(blocks)


def analyze_audio_block(waveform, start, end, sr):
    start_sample = int(start * 16000)
    end_sample = int(end * 16000)
    block_waveform = waveform[:, start_sample:end_sample]

    block_output = process_func_torchaudio(block_waveform, sampling_rate=sr)

    return block_output[0][0],block_output[0][1],block_output[0][2] #Arousal, Valence, Dominance

device = "cuda"
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)

audio_path = '~/Thesis/WavFiles_extraAudioAnalysis_batch3'
id_transcript_path = f'{audio_path}/transcripts/IdentifiedTranscripts'
os.makedirs(f'{audio_path}/transcripts/SpeakerAudio', exist_ok=True)
output_path = f'{audio_path}/transcripts/SpeakerAudio'

for filename in os.listdir(audio_path):
    if (filename.endswith(".WAV") or filename.endswith(".wav")) and not filename.startswith('._'):
        base_name = os.path.splitext(filename)[0]
        
        #Get matching transcript file
        transcript_filename = None
        for transcript_file in os.listdir(id_transcript_path):
            if transcript_file.startswith(base_name) and transcript_file.endswith("identified.csv"):
                transcript_filename = transcript_file
                break
        
        if transcript_filename is not None:
            print(f'Processing {filename}')
            transcript_df = pd.read_csv(os.path.join(id_transcript_path, transcript_filename), sep=';')

            blocks_df = combine_utterances(transcript_df)

            audio_filepath = os.path.join(audio_path, filename)
            waveform, sample_rate = torchaudio.load(audio_filepath)
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

            results = []
            for i, row in blocks_df.iterrows():
                arousal, valence, dominance = analyze_audio_block(waveform, row['start'], row['end'], sr=16000)
                if arousal is not None:
                    results.append({
                        'Sample': i,
                        'Speaker': row['speaker'],
                        'Arousal': arousal,
                        'Dominance': dominance,
                        'Valence': valence,
                        'Start Time': row['start'] * 1000,
                        'End Time': row['end'] * 1000
                    })

            results_df = pd.DataFrame(results)

            output_filename = f'{base_name}_IDAudioAnalysis.csv'
            results_df.to_csv(os.path.join(output_path, output_filename), index=False)

        else:
            print(f"No corresponding transcript found for {filename}")