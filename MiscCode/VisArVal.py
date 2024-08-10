#Arousal valence visualisation without temporal element
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def moving_average(data, window_size):
    return data.rolling(window=window_size).mean()

def plot_arousal_valence_session(df, figname):
    speakers = df['Speaker'].unique()
    speakers = np.delete(speakers, np.where(speakers=='NAN'))
    color_map = {
        'patient': {'valence': 'blue', 'arousal': 'blue'},
        'therapist': {'valence': 'red', 'arousal': 'red'}
    }

    moving_averages = {
        'valence': 'black',
        'arousal': 'black'
    }

    fig, ax = plt.subplots(nrows=2, ncols=len(speakers), figsize=(15, 10), sharex=True)

    for i, speaker in enumerate(speakers):
        speaker_data = df[df['Speaker'] == speaker].copy()
        
        # Calculate moving averages
        speaker_data.loc[:, 'arousal_ma'] = moving_average(speaker_data['Arousal'], window_size=10)
        speaker_data.loc[:, 'valence_ma'] = moving_average(speaker_data['Valence'], window_size=10)
        
        # Plot arousal scores
        ax[0, i].plot(speaker_data['Start Time'], speaker_data['Arousal'], label='Arousal', color=color_map[speaker]['arousal'])
        ax[0, i].plot(speaker_data['Start Time'], speaker_data['arousal_ma'], label='Arousal MA', color=moving_averages['arousal'], linestyle='--')
        ax[0, i].set_title(f'Speaker {speaker}')
        ax[0, i].set_ylabel('Arousal')
        
        # Plot valence scores
        ax[1, i].plot(speaker_data['Start Time'], speaker_data['Valence'], label='Valence', color=color_map[speaker]['valence'])
        ax[1, i].plot(speaker_data['Start Time'], speaker_data['valence_ma'], label='Valence MA', color=moving_averages['valence'], linestyle='--')
        ax[1, i].set_title(f'Speaker {speaker}')
        ax[1, i].set_ylabel('Valence')
        ax[1, i].set_xlabel('Time')

    #ax.legend()
    fig.suptitle(f'Arousal-Valence over time for {figname}')

    return fig, ax

def plot_arousal_valence_model(df, figname):
    fig, ax = plt.subplots(figsize=(10, 10))

    #plot circle
    circle = plt.Circle((0.5, 0.5), 0.5, fill=False)
    ax.add_artist(circle)
    
    #Axes
    ax.axhline(y=0.5, color='gray', linestyle='--')
    ax.axvline(x=0.5, color='gray', linestyle='--')
    
    speakers = df['Speaker'].unique()
    speakers = np.delete(speakers, np.where(speakers=='NAN'))
    color_map = {'patient': 'blue', 'therapist': 'red'}
    
    for speaker in speakers:
        speaker_data = df[df['Speaker'] == speaker]
        color = color_map.get(speaker, 'gray')
        ax.scatter(speaker_data['Valence'], speaker_data['Arousal'], 
                   c=color, alpha=0.6, label=speaker)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')   
    ax.text(0.25, 0.75, 'High Arousal\nNegative Valence', ha='center', va='center')
    ax.text(0.75, 0.75, 'High Arousal\nPositive Valence', ha='center', va='center')
    ax.text(0.25, 0.25, 'Low Arousal\nNegative Valence', ha='center', va='center')
    ax.text(0.75, 0.25, 'Low Arousal\nPositive Valence', ha='center', va='center')
    ax.legend()
    ax.set_title(f'Arousal-Valence Model for {figname}')

    return fig, ax

speaker_arval_path = '~/output/SpeakerSentAnalysis'
fig_path = '~/output/SpeakerSentAnalysis/figures/ArValModel'
session_path = '~/output/SpeakerSentAnalysis/figures/ArValSession'
os.makedirs(fig_path, exist_ok=True)
os.makedirs(session_path, exist_ok=True)

for filename in tqdm(os.listdir(speaker_arval_path)):
    if filename.endswith(".csv"):
        file_path = os.path.join(speaker_arval_path, filename)
        df = pd.read_csv(file_path, delimiter=';')
        figname = filename.replace('.csv', '')

        #plot arousal valence model
        fig, ax = plot_arousal_valence_model(df, figname)
        fig.savefig(os.path.join(fig_path, f'{figname}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        #plot arousal valence over session
        fig2, ax2 = plot_arousal_valence_session(df, figname)
        fig2.savefig(os.path.join(session_path, f'{figname}Session.png'), dpi=300, bbox_inches='tight')
        plt.close(fig2)