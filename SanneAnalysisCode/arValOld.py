# extracts the arousal, valence and dominance from an audio file using the audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim model

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

def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""
    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = torch.from_numpy(y).to(device)
    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()
    return y


import numpy as np
from scipy.io import wavfile
import os
import pandas as pd
from pydub import AudioSegment
segment_length_sec = 300  # 5 minutes
segment_length_samples = int(segment_length_sec * 16000)
millisec_step = 300000
end = millisec_step

device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionModel.from_pretrained(model_name)

folder_path = '~/Thesis/WhisperXTranscriptTest'
os.makedirs('~/Thesis/WhisperXTranscriptTest/temp/', exist_ok=True)
os.makedirs('~/Thesis/WhisperXTranscriptTest/output/', exist_ok=True)
for filename in os.listdir(folder_path):
    if (filename.endswith(".WAV") or filename.endswith(".wav")) and not filename.startswith('._'):
        outputs = []
        wav_file_path = os.path.join(folder_path, filename)
        video_path = wav_file_path
        temp_path = folder_path + '/temp/' + filename
        command = f"ffmpeg -y -i {repr(wav_file_path)} -vn -acodec pcm_s16le -ar 16000 -ac 1 {repr(temp_path)}"
        os.system(command)
        sampling_rate, waveform = wavfile.read(temp_path)
        waveform = np.expand_dims(waveform.astype(np.float32), axis=0)
        #calculate the number of segments for the current WAV file
        num_segments = waveform.shape[1] // segment_length_samples
        audio = AudioSegment.from_wav(temp_path)
        gidx = 0
        start = 0
        segment_data = []

        for i in range(num_segments):
            gidx += 1
            audio[start:end].export(temp_path + str(gidx) + '.wav', format='wav')
            start = end
            end = start + millisec_step
            sampling_rate, waveform = wavfile.read(os.path.join(temp_path + str(gidx) + '.wav'))
            waveform = np.expand_dims(waveform.astype(np.float32), axis=0)
            #process segment
            segment_output = process_func(waveform, sampling_rate)

            outputs.append(segment_output)
            segment_dict = {
            "Sample": i,
            "Arousal": outputs[i][0][0],
            "Dominance": outputs[i][0][1],
            "Valence": outputs[i][0][2],
            "End Time": start
            }
            segment_data.append(segment_dict)

        outputs = np.array(outputs).reshape(num_segments, -1)
        df = pd.DataFrame(segment_data)

        output_csv_file = folder_path + '/output/' + filename + "output.csv" 
        df.to_csv(output_csv_file, index=False)
