#Code to analyse speaker-level utterances using custom RobBERT model for affect as well as RobBERT for sentiment
from transformers import RobertaTokenizer, RobertaForSequenceClassification, pipeline
import torch
import pandas as pd
import traceback
import glob
import os
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_path = '~/ArValWordSent/final_model'
tokenizer_path = '~/ArValWordSent/final_model'

tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
model = RobertaForSequenceClassification.from_pretrained(model_path, use_safetensors=True)

sentiment_model_name = "DTAI-KULeuven/robbert-v2-dutch-sentiment"
sentiment_model = RobertaForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = RobertaTokenizer.from_pretrained(sentiment_model_name)

classifier = pipeline('sentiment-analysis', model=sentiment_model, tokenizer=sentiment_tokenizer)

def predict_arousal_valence(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    predictions = outputs.logits.detach().numpy()[0]
    return predictions[0], predictions[1]

def process_csv_files(df, output_file):
    try:
        segment_data = []
        for _, row in df.iterrows():
            timestamp = row['start']
            end_time = row['end']
            speaker = row['speaker']

            arousal, valence = predict_arousal_valence(row['text'])
            sentiment = classifier(row['text'])
            sentiment_label = sentiment[0]['label']
            
            speaker_row = {
                'Start Time': timestamp,
                'End Time': end_time,
                'Speaker': speaker,
                'Sentiment': sentiment_label,
                'Arousal': arousal,
                'Valence': valence
            }
            segment_data.append(speaker_row)
        
        df_segment = pd.DataFrame(segment_data)
        df_segment.to_csv(output_file, index=False, sep=';')
        print("Speaker-level analysis completed for", output_file)
    except Exception as e:
        print("Error processing DataFrame")
        print("Exception:", str(e))
        traceback.print_exc()

path = "~/output/IdentifiedTranscripts" # path to folder with transcription files 
output_folder = "~/output/SpeakerSentAnalysis" # path to folder where output files are stored

csv_files = glob.glob(os.path.join(path, '*.csv'))

os.makedirs(output_folder, exist_ok=True)

for input_file in csv_files:
    df = pd.read_csv(input_file, delimiter=';')
    output_file_speaker = os.path.join(output_folder, os.path.basename(input_file).replace('.wavoutput.csv_identified.csv', '_SentArVal.csv'))
    process_csv_files(df, output_file_speaker)