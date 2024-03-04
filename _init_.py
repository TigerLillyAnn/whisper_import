import numpy as np
import whisper
import time
import pyaudio
import string
import socket

# Constants and settings
CHUNK = 1024
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
DEVICE_INDEX = None
RECORD_SECONDS = 8
SAMPLES_PER_WINDOW = RECORD_SECONDS * RATE
key_phrase = "hey fetch"

# Load  Whisper model
model = whisper.load_model("small")

# Open the microphone
audio_interface = pyaudio.PyAudio()
stream = audio_interface.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              input_device_index=DEVICE_INDEX,
                              frames_per_buffer=CHUNK)

# Process and transcribe audio data
def process_audio(audio_data):
    audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)*(1/32768.0)
    audio = audio.astype(np.float32)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)

    return result.text

port = 12345

# Main functionality for real-time audio transcription
print("Starting transcription...")
try:
    while True:
        audio_data = b""
        start_time = time.time()

        # Collect 5 seconds of audio
        while time.time() - start_time < RECORD_SECONDS:
            data = stream.read(CHUNK, exception_on_overflow= False)
            audio_data += data

        # Process and transcribe the audio
        transcript = process_audio(audio_data)
        print("Transcript:", transcript)
        transcript = transcript.translate(str.maketrans('', '', string.punctuation)).lower()
        # print("Transcript modified:", transcript)
        if key_phrase in transcript:
            s = socket.socket()
            s.connect(('152.23.89.159', port))
            s.send(transcript.split(key_phrase)[1].encode("utf-8"))
            s.close()

            print("Transcript:", transcript.split(key_phrase)[1])

except KeyboardInterrupt:
    print("Stopped real-time transcription.")

# Close the microphone
stream.stop_stream()
stream.close()
audio_interface.terminate()