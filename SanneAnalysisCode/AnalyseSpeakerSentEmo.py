#Conversion of previous sentiment emotion extraction from text method. Instead of analysing 5 minute chunks of all text,
#text from each speaker is grouped and analysed, with a max of 4 minutes, matching the audio analysis process

import csv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
import pandas as pd
import glob
import os
from collections import Counter
import traceback
import torch
from tqdm import tqdm

nltk.download('punkt')
import argostranslate.package
import argostranslate.translate

from_code = "nl"
to_code = "en"

# Download and install Argos Translate package
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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#Sentiment
sent_model_name = "DTAI-KULeuven/robbert-v2-dutch-sentiment"
sent_model = RobertaForSequenceClassification.from_pretrained(sent_model_name)
sent_tokenizer = RobertaTokenizer.from_pretrained(sent_model_name)
sent_classifier = pipeline('sentiment-analysis', model=sent_model, tokenizer=sent_tokenizer)

#Specific emotion from EmoRoberta
emo_tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
emo_model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

#Arousal Valence
arval_tokenizer = RobertaTokenizer.from_pretrained('~/Thesis/RobBERT/ArValWordSent/final_model')
arval_model = RobertaForSequenceClassification.from_pretrained('~/Thesis/RobBERT/ArValWordSent/final_model', use_safetensors=True)

def translate_text(text, from_lang, to_lang):
    translation = from_lang.get_translation(to_lang)
    translated_text = translation.translate(text)
    return translated_text

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

def predict_arousal_valence(text):
    inputs = arval_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = arval_model(**inputs)
    predictions = outputs.logits.detach().numpy()[0]
    return predictions[0], predictions[1]

def analyze_sentences(text, classifier, emotion, from_lang, to_lang):
    sentences = nltk.sent_tokenize(text)
    sentiment_counter = Counter()
    emotion_counter = Counter()
    
    for sentence in sentences:
        #sentiment analysis
        sentiment_result = classifier(sentence)
        sentiment_counter.update([sentiment_result[0]['label']])

        #translate then emotion analysis
        translated_sentence = translate_text(sentence, from_lang, to_lang)
        emotion_result = emotion(translated_sentence)
        emotions = [emotion['label'] for emotion in emotion_result]
        emotion_counter.update(emotions)
    
    return sentiment_counter, emotion_counter

def process_csv_files(df, output_file, from_lang, to_lang):
    try:
        # Process each row in the input DataFrame
        segment_data = []
        for _, row in df.iterrows():
            timestamp = row['start'] * 1000
            end_time = row['end'] * 1000
            speaker = row['speaker']
            dutch_transcript = row['text']

            #Get arousal, valence
            arousal, valence = predict_arousal_valence(dutch_transcript)
            
            # Analyze sentences in the transcript
            sentiment_counter, emotion_counter = analyze_sentences(dutch_transcript, sent_classifier, emotion, from_lang, to_lang)
            
            # Combine sentiment and emotion counts
            combined_counts = {**sentiment_counter, **emotion_counter}
            
            # Create a dictionary for the row data
            speaker_row = {
                'Start Time': timestamp,
                'End Time': end_time,
                'Speaker': speaker,
                'Arousal': arousal,
                'Valence': valence,
                **combined_counts
            }
            segment_data.append(speaker_row)
        
        # Write the row-level analysis to the output CSV file
        df_segment = pd.DataFrame(segment_data)
        df_segment.to_csv(output_file, index=False)
        # Print a message indicating the completion of the analysis for the current file
        print("Speaker-level analysis completed for", output_file)
    except Exception as e:
        # Print the exception message
        print("Error processing DataFrame")
        print("Exception:", str(e))
        traceback.print_exc()

paths = ['~/Thesis/WavFiles_extraAudioAnalysis/transcripts/IdentifiedTranscripts',
         '~/Thesis/WavFiles_extraAudioAnalysis_batch2/transcripts/IdentifiedTranscripts',
         '~/Thesis/WavFiles_extraAudioAnalysis_batch3/transcripts/IdentifiedTranscripts']

for path in paths:
    csv_files = glob.glob(os.path.join(path, '*.csv'))
    parts = path.split('/')
    parts[-1] = 'SpeakerTextEmo'
    output_folder = '/'.join(parts)
    os.makedirs(output_folder, exist_ok=True)

    for input_file in tqdm(csv_files):
        df = pd.read_csv(input_file, delimiter=';')
        speaker_level = combine_utterances(df)
        output_file_speaker = os.path.join(output_folder, os.path.basename(input_file).replace('.csv', '_speakerEmo.csv'))
        process_csv_files(speaker_level, output_file_speaker, from_lang, to_lang)