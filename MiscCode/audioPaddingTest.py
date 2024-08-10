#Audio padding test
from pydub import AudioSegment
import os

# Load your existing WAV file
audio = AudioSegment.from_file("~/Thesis/WhisperXTranscriptTest/test.wav")

# Create a 5-second silence audio segment
silence = AudioSegment.silent(duration=5000)  # duration in milliseconds

# Add silence to the beginning of the original audio
padded_audio = silence + audio

# Export the modified file
padded_audio.export("~/Thesis/WhisperXTranscriptTest/testout.wav", format="wav")