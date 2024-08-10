#Multithreaded attempt at audio analysis
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
from threading import Thread, Lock

millisec_step = 300000
segment_length_torch = 1 * 60 * 16000
sample_to_milsec = 1000 / 16000

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
        print("torch.no_grad")
        model.eval()
        print("model eval")
        #torch.cuda.empty_cache()
        output = model(input_values)[0 if embeddings else 1]
        print("output")
    output_array = output.detach().cpu().numpy()
    return output_array
    
def process_chunk(chunk, sampling_rate, lock):
    """Process a chunk of audio data."""
    with torch.no_grad():
        model.eval()
        output = process_func_torchaudio(chunk, sampling_rate)[0]
    lock.acquire()
    # Add the output to a thread-safe data structure (e.g., list)
    outputs.append(output)
    lock.release()


def make_segment_dict(sample_nr, data, end_time):
    segment_dict = {
                "Sample": sample_nr,  # Adjust indexing based on processing
                "Arousal": data[0],  # Assuming arousal is first value
                "Dominance": data[1],  # Assuming dominance is second
                "Valence": data[2],  # Assuming valence is third
                "End Time": end_time,
            }
    return segment_dict


device = "cpu"
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name).to(device)

folder_path = '~/Thesis/WhisperXTranscriptTest'
os.makedirs('~/WhisperXTranscriptTest/temp/', exist_ok=True)
os.makedirs('~/WhisperXTranscriptTest/output/', exist_ok=True)

ebot = EmailBot()  # For email error notifications

try:
    for filename in os.listdir(folder_path):
        if (filename.endswith(".WAV") or filename.endswith(".wav")) and not filename.startswith('._'):
            print(f"Processing: {filename}")
            outputs = []
            lock = Lock()
            wav_file_path = os.path.join(folder_path, filename)
            waveform, sampling_rate = torchaudio.load(wav_file_path)
            waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
                waveform = resampler(waveform)
                sampling_rate = 16000

            #process in chunks
            segment_data = []
            start_time = 0
            end_time = segment_length_torch  #initial window
            print("Sampling rate", sampling_rate)

            #multithread chunks
            num_threads = 4 
            threads = []
            start_time = 0
            end_time = segment_length_torch
            while end_time <= waveform.shape[1]:
                chunk = waveform[:, start_time:end_time]
                #create thread
                if len(threads) < num_threads:
                    thread = Thread(target=process_chunk, args=(chunk, sampling_rate, lock))
                    threads.append(thread)
                    thread.start()
                else:
                    #wait for a thread to finish before creating a new one
                    for thread in threads:
                        thread.join()
                        if not thread.is_alive():
                            threads.remove(thread)
                            break
                start_time = end_time
                end_time += segment_length_torch

            for thread in threads:
                thread.join()

            #combine thread results
            segment_data = []
            for output in outputs:
                print(output)
                segment_dict = make_segment_dict(len(segment_data), output, end_time * sample_to_milsec)
                segment_data.append(segment_dict)

            df = pd.DataFrame(segment_data)
            output_csv_file = folder_path + '/output/' + filename + "output.csv"
            df.to_csv(output_csv_file, index=False)
except Exception:
    ebot.setEmailContent(traceback.format_exc())
    ebot.sendEmail()