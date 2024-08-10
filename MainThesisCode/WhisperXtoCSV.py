#Main WhisperX file which converts Wav audio to transcripts
import whisperx
import csv
import torch
import os
import traceback
import json
import pydub

from emailMonitor import EmailBot

def dict_to_csv(data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(['', 'start', 'end', 'text', 'words', 'speaker'])

        for row_num, segment in enumerate(data['segments'], start=1):
            if 'speaker' not in segment.keys():
                segment['speaker'] = 'NAN'
            words_json = json.dumps(segment['words'])
            writer.writerow([row_num, segment['start'], segment['end'], segment['text'], words_json, segment['speaker']])


device = "cpu" # needs to change for uni computer - GPU should be used
batch_size = 5 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)
options = {
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
    }

folder_path = '~/Thesis/WavFiles_extraAudioAnalysis_batch3'
os.makedirs('~/Thesis/WavFiles_extraAudioAnalysis_batch3/transcripts/', exist_ok=True)

ebot = EmailBot() #For email error notifications

try:
    for filename in os.listdir(folder_path):
        if (filename.endswith(".WAV") or filename.endswith(".wav")) and not filename.startswith('._'):
            print(f"Processing {filename}")
            file = os.path.join(folder_path, filename)

            model = whisperx.load_model("large-v3", device, compute_type=compute_type, language='nl', asr_options=options) # nl for dutch

            audio = whisperx.load_audio(file)
            result = model.transcribe(audio, batch_size=batch_size)

            # align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            print(result)

            # diarization
            diarize_model = whisperx.DiarizationPipeline(use_auth_token='', device=device) #Auth-token from Huggingface
            diarize_segments = diarize_model(audio, max_speakers=2)

            result = whisperx.assign_word_speakers(diarize_segments, result)

            new_filename = folder_path + '/transcripts/' + filename + "output.csv"
            print(result)
            dict_to_csv(result, new_filename)

except Exception:
    ebot.setEmailContent(traceback.format_exc())
    ebot.sendEmail()