#Script for visualising a number of features such as emotion, affect and sentiment
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob

emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
colors = {'anger': 'red', 'disgust': 'green', 'fear': 'purple', 
          'joy': 'yellow', 'sadness': 'blue', 'surprise': 'orange'}

def plot_affect_emotions_for_sentences(df, filename, directory):
    base_dir = f'{directory}/Figures'
    os.makedirs(base_dir, exist_ok=True)
    subdirs = ["Affect", "AffectOverTime", "EmotionsOverTime", "AffectDistribution", "EmotionComposition"]
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    base_filename = base_filename.replace('.wavoutput_speakerEmo.csv_identified_speakerEmo', '')
    
    #Create emotion columns if missing for easier plot handling
    for emotion in emotions:
        if emotion not in df.columns:
            df[emotion] = 0
    
    df_filled = df[emotions + ['Arousal', 'Valence', 'Speaker', 'Start Time']].fillna(0)
    
    #Affect circumplex plot
    fig, ax = plt.subplots(figsize=(15, 12))
    
    scatter = ax.scatter(df_filled['Valence'], df_filled['Arousal'], alpha=0.5, c=df_filled['Start Time'], cmap='viridis')
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Time Progression', rotation=270, labelpad=15)

    circle1 = plt.Circle((0.5, 0.5), 0.25, fill=False, color='gray')
    circle2 = plt.Circle((0.5, 0.5), 0.5, fill=False, color='gray')
    ax.add_artist(circle1)
    ax.add_artist(circle2)

    ax.axhline(y=0.5, color='k', linestyle='--')
    ax.axvline(x=0.5, color='k', linestyle='--')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Valence', fontsize=12)
    ax.set_ylabel('Arousal', fontsize=12)
    ax.set_title(f'Arousal-Valence Circumplex - {base_filename}', fontsize=14)

    ax.text(0.85, 0.85, 'High Arousal\nPositive Valence', ha='center', va='center', fontsize=10)
    ax.text(0.15, 0.85, 'High Arousal\nNegative Valence', ha='center', va='center', fontsize=10)
    ax.text(0.15, 0.15, 'Low Arousal\nNegative Valence', ha='center', va='center', fontsize=10)
    ax.text(0.85, 0.15, 'Low Arousal\nPositive Valence', ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "Affect", f"{base_filename}_circumplex.png"))
    plt.close()

    #Affect over time plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    therapist_data = df_filled[df_filled['Speaker'] == 'therapist']
    patient_data = df_filled[df_filled['Speaker'] == 'patient']
    
    ax1.plot(therapist_data['Start Time'], therapist_data['Arousal'], color='red', label='Arousal')
    ax1.plot(therapist_data['Start Time'], therapist_data['Valence'], color='blue', label='Valence')
    ax1.set_ylabel('Affect Level')
    ax1.set_title(f'Therapist Arousal and Valence Over Time - {base_filename}')
    ax1.legend()
    
    ax2.plot(patient_data['Start Time'], patient_data['Arousal'], color='red', label='Arousal')
    ax2.plot(patient_data['Start Time'], patient_data['Valence'], color='blue', label='Valence')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Affect Level')
    ax2.set_title(f'Patient Arousal and Valence Over Time - {base_filename}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "AffectOverTime", f"{base_filename}_affect_over_time.png"))
    plt.close()

    #Emotion over time plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    for emotion, color in colors.items():
        ax1.plot(therapist_data['Start Time'], therapist_data[emotion], color=color, label=emotion)
    ax1.set_ylabel('Emotion Count')
    ax1.set_title(f'Therapist Emotion Count Over Time - {base_filename}')
    ax1.legend()
    
    for emotion, color in colors.items():
        ax2.plot(patient_data['Start Time'], patient_data[emotion], color=color, label=emotion)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Emotion Count')
    ax2.set_title(f'Patient Emotion Count Over Time - {base_filename}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "EmotionsOverTime", f"{base_filename}_emotions_over_time.png"))
    plt.close()

    #Affect distribution plots
    plt.figure(figsize=(10, 6))
    df_melt = pd.melt(df_filled, id_vars=['Speaker'], value_vars=['Arousal', 'Valence'], var_name='affect', value_name='level')
    sns.boxplot(x='Speaker', y='level', hue='affect', data=df_melt)
    plt.title(f'Distribution of Affect by Speaker - {base_filename}')
    plt.savefig(os.path.join(base_dir, "AffectDistribution", f"{base_filename}_affect_distribution.png"))
    plt.close()

    #Emotion composition plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    therapist_emotions = df_filled[df_filled['Speaker'] == 'therapist'][emotions].sum().fillna(0)
    patient_emotions = df_filled[df_filled['Speaker'] == 'patient'][emotions].sum().fillna(0)
    
    if therapist_emotions.sum() > 0:
        ax1.pie(therapist_emotions, labels=emotions, autopct='%1.1f%%', startangle=90, colors=colors.values())
        ax1.set_title(f'Therapist Emotion Composition - {base_filename}')
    else:
        ax1.text(0.5, 0.5, 'No emotion data', ha='center', va='center')
        ax1.set_title(f'Therapist Emotion Composition - {base_filename} (No Data)')
    
    if patient_emotions.sum() > 0:
        ax2.pie(patient_emotions, labels=emotions, autopct='%1.1f%%', startangle=90, colors=colors.values())
        ax2.set_title(f'Patient Emotion Composition - {base_filename}')
    else:
        ax2.text(0.5, 0.5, 'No emotion data', ha='center', va='center')
        ax2.set_title(f'Patient Emotion Composition - {base_filename} (No Data)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "EmotionComposition", f"{base_filename}_emotion_composition.png"))
    plt.close()

def process_csv_files(directory):
    for csv_file in glob.glob(os.path.join(directory, '*.csv')):
        print(csv_file)
        df = pd.read_csv(csv_file)
        plot_affect_emotions_for_sentences(df, csv_file, directory)

directory = '~/Thesis/WavFiles_extraAudioAnalysis/transcripts/SpeakerTextEmo'
process_csv_files(directory)