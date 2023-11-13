import whisper

if __name__ == "__main__":
    model = whisper.load_model('base')

    audio = whisper.load_audio('test.mp3') #insert audio to test
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    print(result.text)