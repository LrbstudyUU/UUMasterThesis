#Helper script that creates a large dataframe of transcript names and their individual texts, preparing for LIWC
import os
import pandas as pd
import re

def combine_transcripts(folder_path, output_file_split, output_file_full):
    combined_data_split = []
    combined_data_full = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath, delimiter=';')
            
            grouped = df.groupby('speaker')['text'].agg(' '.join).reset_index()
            grouped['transcript_name'] = filename
            combined_data_split.append(grouped[['transcript_name', 'speaker', 'text']])

            full_text = ' '.join(df['text'])
            combined_data_full.append({
                'transcript_name': filename,
                'text': full_text
            })

    result_split = pd.concat(combined_data_split, ignore_index=True)
    result_split.to_csv(output_file_split, index=False)

    result_full = pd.DataFrame(combined_data_full)
    result_full.to_csv(output_file_full, index=False)

def combine_csv_files(file_paths, output_file):
    combined_df = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved as {output_file}")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

def prepare_for_liwc(input_file, output_file):
    df = pd.read_csv(input_file)
    df['text'] = df['text'].apply(clean_text)
    df.to_csv(output_file, index=False)
    print(f"Cleaned data saved as {output_file}")

#BELOW is for cleaning csvs for LIWC
input_file = '~/transcripts_split.csv'
output_file = '~/cleaned_split.csv'
prepare_for_liwc(input_file, output_file)