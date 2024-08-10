#Code to extract sentiment and emotion as well as affect with custom RobBERT model from text, adapted from Rivka's code, translation using argos added
import csv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import RobertaModel, RobertaTokenizer, RobertaForSequenceClassification, RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
import pandas as pd
import glob
import os
from collections import Counter
import traceback
import torch
import argostranslate.package
import argostranslate.translate
from tqdm import tqdm

from_code = "nl"
to_code = "en"

#Download and install Argos Translate package
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())

installed_languages = argostranslate.translate.get_installed_languages()
from_lang = next(filter(lambda x: x.code == "nl", installed_languages), None)
to_lang = next(filter(lambda x: x.code == "en", installed_languages), None)
if not from_lang or not to_lang:
    raise Exception("Required package missing")

def translate_text(text, from_lang, to_lang):
    translation = from_lang.get_translation(to_lang)
    translated_text = translation.translate(text)
    return translated_text

#sentiment model
sentiment_model_name = "DTAI-KULeuven/robbert-v2-dutch-sentiment"
sentiment_model = RobertaForSequenceClassification.from_pretrained(sentiment_model_name)
tokenizer = RobertaTokenizer.from_pretrained(sentiment_model_name)
classifier = pipeline('sentiment-analysis', model=sentiment_model, tokenizer=tokenizer)

#emotion model
emo_tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
emo_model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

#Custom arousal Valence model
arval_tokenizer = RobertaTokenizer.from_pretrained('~/Thesis/RobBERT/ArValWordSent/final_model') #Custom RobBERT model
arval_model = RobertaForSequenceClassification.from_pretrained('~/Thesis/RobBERT/ArValWordSent/final_model', use_safetensors=True)

def predict_arousal_valence(text):
    inputs = arval_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = arval_model(**inputs)
    predictions = outputs.logits.detach().numpy()[0]
    return predictions[0], predictions[1]

def process_csv_files(csv_files, output_folder):
    for input_file in tqdm(csv_files):
        try:
            print(input_file)
            output_file_segment = os.path.join(output_folder, os.path.basename(input_file).replace('.csv', '_segment.csv'))

            with open(input_file, 'r', newline='') as csv_input:
                reader = csv.DictReader(csv_input, delimiter=';')
                #Process each row in the input CSV file
                segment_data = []
                current_start_time = None
                current_end_time = None
                current_speaker = None
                emotion_counter = Counter()
                arousal_levels = []
                valence_levels = []
                for row in reader:
                    timestamp = float(row['start']) * 1000
                    speaker = row['speaker']
                    dutch_transcript = row['text']
                    if current_start_time is None:
                        current_start_time = timestamp
                        current_end_time = timestamp
                        current_speaker = speaker
                    #sentiment analysis
                    scores_sentiment = classifier(dutch_transcript)
                    #emotion extraction
                    translated_text = translate_text(dutch_transcript, from_lang, to_lang)
                    emotions = emotion(translated_text)
                    emotions_str = ', '.join([emotion['label'] for emotion in emotions])
                    #arousal prediction
                    arousal, valence = predict_arousal_valence(dutch_transcript)
                    arousal_levels.append(arousal)
                    valence_levels.append(valence)
                    #update sentiment counter
                    sentiment_label = scores_sentiment[0]['label']
                    emotion_counter.update(emotions_str.split(', '))
                    #check if current segment is past 5 minutes
                    if timestamp >= current_start_time + 5 * 60 * 1000:
                        avg_arousal_level = sum(arousal_levels) / len(arousal_levels) if arousal_levels else 0
                        avg_valence_level = sum(valence_levels) / len(valence_levels) if valence_levels else 0
                        segment_row = {
                            'Start Time': current_start_time,
                            'End Time': current_end_time,
                            'Speaker': current_speaker,
                            'Sentiment': sentiment_label,
                            'Arousal': avg_arousal_level,
                            'Valence': avg_valence_level,
                            **emotion_counter
                        }
                        segment_data.append(segment_row)
                        #Reset the variables next segment
                        current_start_time = timestamp
                        current_end_time = timestamp
                        current_speaker = speaker
                        emotion_counter = Counter()
                        arousal_levels = []
                        valence_levels = []
                    else:
                        #Update the end time for current segment
                        current_end_time = timestamp
                avg_arousal_level = sum(arousal_levels) / len(arousal_levels) if arousal_levels else 0
                avg_valence_level = sum(valence_levels) / len(valence_levels) if valence_levels else 0
                segment_row = {
                    'Start Time': current_start_time,
                    'End Time': current_end_time,
                    'Speaker': current_speaker,
                    'Sentiment': sentiment_label,
                    'Arousal': avg_arousal_level,
                    'Valence': avg_valence_level,
                    **emotion_counter
                }
                segment_data.append(segment_row)

            df_segment = pd.DataFrame(segment_data)
            df_segment.to_csv(os.path.join(output_folder, output_file_segment), index=False)
            print("Segment-level analysis completed for", input_file)
        except Exception as e:
            print("Error processing file:", input_file)
            print("Exception:", str(e))
            traceback.print_exc()

paths = ['~/Thesis/WavFiles_extraAudioAnalysis/transcripts',
         '~/Thesis/WavFiles_extraAudioAnalysis_batch2/transcripts',
         '~/Thesis/WavFiles_extraAudioAnalysis_batch3/transcripts']

for path in paths:
    csv_files = glob.glob(os.path.join(path, '*.csv'))
    parts = path.split('/')
    parts[-1] = 'arousalSegmentsNoID' 
    output_folder = '/'.join(parts)
    os.makedirs(output_folder, exist_ok=True)
    process_csv_files(csv_files, output_folder)