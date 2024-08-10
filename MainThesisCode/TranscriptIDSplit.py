#Helper script to assign "patient" "therapist" identities to transcripts based on keyword matching
import pandas as pd
import os

def identify_therapist_patient(csv_file):
    df = pd.read_csv(csv_file, sep=";")
    therapist_indicators = ['?', 'sessie'] #Main identifiers of therapist, might need adjustment but questions seem to be a good indicator

    potential_therapists = {}
    #Search through document, count number of indicators for each speaker
    for index, row in df.iterrows():
        speaker = row['speaker']
        text = row['text'].lower()
        
        if any(indicator in text for indicator in therapist_indicators):
            if speaker in potential_therapists:
                potential_therapists[speaker] += 1
            else:
                potential_therapists[speaker] = 1

    #Identify speakers based on indicator counts
    if potential_therapists:
        therapist_id = max(potential_therapists, key=potential_therapists.get)
        patient_id = [speaker for speaker in df['speaker'].unique() if speaker != therapist_id and speaker != 'NAN']

        if len(patient_id) == 1:
            patient_id = patient_id[0]
        else:
            patient_id = None
            print("Warning: Unable to uniquely identify the patient ID.")
    else:
        therapist_id = None
        patient_id = None
        print("Warning: Unable to identify the therapist.")

    #Divide transcripts
    if therapist_id and patient_id:
        therapist_transcript = df[df['speaker'] == therapist_id]
        patient_transcript = df[df['speaker'] == patient_id]
        id_transcript = df.copy()
        id_transcript['speaker'] = id_transcript['speaker'].replace({therapist_id: 'therapist', patient_id: 'patient'})
    else:
        therapist_transcript = None
        patient_transcript = None
        id_transcript = None

    return therapist_transcript, patient_transcript, id_transcript #, therapist_id, patient_id

transcript_path =  "~/WavFiles_extraAudioAnalysis_batch3/transcripts"
#thresholded_df = pd.read_csv("~/filtered_dataframe.csv")

os.makedirs(f'{transcript_path}/IdentifiedTranscripts', exist_ok=True)
os.makedirs(f'{transcript_path}/Therapist', exist_ok=True)
os.makedirs(f'{transcript_path}/Patient', exist_ok=True)

id_transcript_path = f"{transcript_path}/IdentifiedTranscripts"
therapist_path = f"{transcript_path}/Therapist"
patient_path = f"{transcript_path}/Patient"

error_files = []
for filename in os.listdir(transcript_path):
    if filename.endswith(".csv") and not filename.startswith('._'):
        file_path = os.path.join(transcript_path, filename)
        therapist_transcript, patient_transcript, id_transcript = identify_therapist_patient(file_path)
        if therapist_transcript is not None and patient_transcript is not None:
            id_transcript.to_csv(f"{id_transcript_path}/{filename}_identified.csv", sep=";")
            therapist_transcript.to_csv(f"{therapist_path}/{filename}_therapist.csv", sep=";")
            patient_transcript.to_csv(f"{patient_path}/{filename}_patient.csv", sep=";")
        else:
            error_files.append(filename)

#This is a modified loop to work with a thresholded list of transcripts from GetSpeakerAvg.ipynb
# for filename in thresholded_df['transcript_id']:
#     therapist_transcript, patient_transcript, id_transcript = identify_therapist_patient(filename)
#     if therapist_transcript is not None and patient_transcript is not None:
#         id_transcript.to_csv(f"{id_transcript_path}/{os.path.basename(filename)}_identified.csv", sep=";")
#         therapist_transcript.to_csv(f"{therapist_path}/{os.path.basename(filename)}_therapist.csv", sep=";")
#         patient_transcript.to_csv(f"{patient_path}/{os.path.basename(filename)}_patient.csv", sep=";")
#     else:
#         error_files.append(filename)

#These error files exist when no distinction can be made
print(error_files)