#Visualisations of affect in different forms
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob

def plot_affect_emotions_for_sentences(df, filename, directory):
    base_dir = f'{directory}/Figures'
    os.makedirs(base_dir, exist_ok=True)
    subdirs = ["Affect", "AffectOverTime", "EmotionsOverTime", "AffectDistribution", "EmotionComposition"]
    for subdir in subdirs:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    base_filename = base_filename.replace('.wavoutput_speakerEmo.csv_identified_speakerEmo', '')
    
    #Affect circumplex plot
    for speaker in ['therapist', 'patient']:
        fig, ax = plt.subplots(figsize=(12, 10))
        speaker_data = df[df['Speaker'] == speaker]
        scatter = ax.scatter(speaker_data['Valence'], speaker_data['Arousal'], alpha=0.5, c=speaker_data['Start Time'], cmap='viridis')
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Time Progression (s)', rotation=270, labelpad=15)
        
        circle1 = plt.Circle((0.5, 0.5), 0.25, fill=False, color='gray')
        circle2 = plt.Circle((0.5, 0.5), 0.5, fill=False, color='gray')
        ax.add_artist(circle1)
        ax.add_artist(circle2)
        
        ax.axhline(y=0.5, color='k', linestyle='--')
        ax.axvline(x=0.5, color='k', linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Valence', fontsize=14)
        ax.set_ylabel('Arousal', fontsize=14)
        ax.set_title(f'{speaker.capitalize()} Arousal-Valence Circumplex - {base_filename}', fontsize=14)
        
        ax.text(0.85, 0.85, 'High Arousal\nPositive Valence', ha='center', va='center', fontsize=14)
        ax.text(0.15, 0.85, 'High Arousal\nNegative Valence', ha='center', va='center', fontsize=14)
        ax.text(0.15, 0.15, 'Low Arousal\nNegative Valence', ha='center', va='center', fontsize=14)
        ax.text(0.85, 0.15, 'Low Arousal\nPositive Valence', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "Affect", f"{base_filename}_{speaker}_circumplex.png"))
        plt.close()

    #Affect over time plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    therapist_data = df[df['Speaker'] == 'therapist']
    patient_data = df[df['Speaker'] == 'patient']
    
    ax1.plot(therapist_data['Start Time'], therapist_data['Arousal'], color='red', label='Arousal')
    ax1.plot(therapist_data['Start Time'], therapist_data['Valence'], color='blue', label='Valence')
    ax1.set_ylabel('Affect Level')
    ax1.set_title(f'Therapist Arousal and Valence Over Time - {base_filename}')
    ax1.legend()
    
    ax2.plot(patient_data['Start Time'], patient_data['Arousal'], color='red', label='Arousal')
    ax2.plot(patient_data['Start Time'], patient_data['Valence'], color='blue', label='Valence')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Affect Level')
    ax2.set_title(f'Patient Arousal and Valence Over Time - {base_filename}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "AffectOverTime", f"{base_filename}_affect_over_time.png"))
    plt.close()

    #Affect distribution plots
    plt.figure(figsize=(10, 6))
    df_melt = pd.melt(df, id_vars=['Speaker'], value_vars=['Arousal', 'Valence'], var_name='affect', value_name='level')
    sns.boxplot(x='Speaker', y='level', hue='affect', data=df_melt)
    plt.title(f'Distribution of Affect by Speaker - {base_filename}')
    plt.savefig(os.path.join(base_dir, "AffectDistribution", f"{base_filename}_affect_distribution.png"))
    plt.close()

def process_csv_files(directory):
    for csv_file in glob.glob(os.path.join(directory, '*.csv')):
        print(csv_file)
        df = pd.read_csv(csv_file, delimiter=';')
        plot_affect_emotions_for_sentences(df, csv_file, directory)

directory = '~/output/SpeakerSentAnalysis'
process_csv_files(directory)