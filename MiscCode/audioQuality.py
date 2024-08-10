#Audio quality analysis tests
import librosa
import numpy as np
from scipy.spatial.distance import cosine
#import sounddevice as sd

def extract_segment_features(audio_path, start_sec, end_sec, play_sound=False):
    y, sr = librosa.load(audio_path, sr=None)
    start_sample = int(start_sec * sr)
    print(start_sample)
    end_sample = int(end_sec * sr)
    segment = y[start_sample:end_sample]

    # if play_sound:
    #     sd.play(segment, sr)
    #     sd.wait()  # Wait until the audio has finished playing

    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

def compare_speaker_segments(audio_path, start1, end1, start2, end2):
    mfccs1 = extract_segment_features(audio_path, start1, end1)
    mfccs2 = extract_segment_features(audio_path, start2, end2)
    similarity = 1 - cosine(mfccs1, mfccs2)
    return similarity

# def SNR_and_Spectral(audio_path):
#     audio, sr = librosa.load('~/Thesis/LennardTranscript_Batch1/test.wav', sr=16000)

#     #Get power spectrum
#     S = np.abs(librosa.stft(audio))

#     #Get SNR
#     signal_power = np.mean(audio**2)
#     noise_power = np.mean((audio - np.mean(audio))**2)
#     snr = 10 * np.log10(signal_power / noise_power)

#     #Get spectral flatness
#     sfm = librosa.feature.spectral_flatness(S=S)

#     print(f"SNR: {snr} dB")
#     print(f"Spectral Flatness: {np.mean(sfm)}")

audio_path = "~/Thesis/LennardTranscript_Batch1/session1.wav"
start1, end1 = 387.500, 390.000  #Speaker0 numbers in seconds
start2, end2 = 417.000, 422.500  #Speaker1
    
audio_path2 = "~/Thesis/LennardTranscript_Batch1/session2.wav"
start1_2, end1_2 = 54.500, 58.073
start2_2, end2_2 = 101.000, 105.060

similarity_score = compare_speaker_segments(audio_path, start1, end1, start2, end2)
print("Similarity Score:", similarity_score)

sim_score = compare_speaker_segments(audio_path2, start1_2, end1_2, start2_2, end2_2)
print("Similarity score 2:", sim_score)