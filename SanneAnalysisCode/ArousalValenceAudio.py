# extracts the arousal, valence and dominance from an audio file using the audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim model
from emailMonitor import EmailBot

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
    input_values = processor(waveform.squeeze(), sampling_rate=sampling_rate, return_tensors="pt", padding=True).input_values.to(device)
    with torch.no_grad():
        model.eval()
        print("model eval")
        #torch.cuda.empty_cache()
        output = model(input_values)[0 if embeddings else 1]
    output_array = output.detach().cpu().numpy()
    return output_array

def make_segment_dict(sample_nr, data, end_time):
    segment_dict = {
                "Sample": sample_nr,  # Adjust indexing based on processing
                "Arousal": data[0][0],  # Assuming arousal is first value
                "Dominance": data[0][1],  # Assuming dominance is second
                "Valence": data[0][2],  # Assuming valence is third
                "End Time": end_time,
            }
    return segment_dict


import numpy as np
from scipy.io import wavfile
import os
import pandas as pd
from pydub import AudioSegment
segment_length_sec = 300  # 5 minutes
segment_length_samples = int(segment_length_sec * 16000)
millisec_step = 300000
segment_length_torch = 5*60*16000
sample_to_milsec = 1000/16000

# load model from hub
device = "cpu"
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)

folder_path = '~/Thesis/WavFiles_extraAudioAnalysis_batch3'
os.makedirs('~Thesis/WavFiles_extraAudioAnalysis_batch3/output/', exist_ok=True)

ebot = EmailBot() #For email error notifications

try:
    for filename in os.listdir(folder_path):
        if (filename.endswith(".WAV") or filename.endswith(".wav")) and not filename.startswith('._'):
            print(f"Processing: {filename}")
            outputs = []
            wav_file_path = os.path.join(folder_path, filename)
            waveform, sampling_rate = torchaudio.load(wav_file_path)
            waveform = torch.mean(waveform, dim=0, keepdim=True) # Convert to mono

            if sampling_rate != 16000: # Resample to 16000 because wav2vec2 is trained in 16000
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
                waveform = resampler(waveform)
                sampling_rate = 16000

            # Process audio in chunks
            segment_data = []
            start_time = 0
            end_time = segment_length_torch  # Initial window
            print("Sampling rate", sampling_rate)

            while end_time <= waveform.shape[1]:
                print(f"Processing chunk {len(segment_data)}")
                chunk = waveform[:, start_time:end_time]  # Get chunk of waveform
                segment_output = process_func_torchaudio(chunk, sampling_rate) # Process chunk

                segment_dict = make_segment_dict(len(segment_data), segment_output, end_time*sample_to_milsec)
                segment_data.append(segment_dict)

                start_time = end_time
                end_time += segment_length_torch  # Slide window
            
            if start_time < waveform.size(1):
                final_chunk = waveform[:, start_time:]
                segment_output = process_func_torchaudio(final_chunk, sampling_rate)
                segment_dict = make_segment_dict(len(segment_data), segment_output, end_time*sample_to_milsec)
                segment_data.append(segment_dict)

            df = pd.DataFrame(segment_data)
            output_csv_file = folder_path + '/output/' + filename + "output.csv"
            df.to_csv(output_csv_file, index=False)
            
except Exception:
    ebot.setEmailContent(traceback.format_exc())
    ebot.sendEmail()