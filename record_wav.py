#record data through the soundcard and save as signal.wav

import pyaudio
import wave

CHUNK = 441
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100	#sampling rate
RECORD_SECONDS = 30	#length of recording
WAVE_OUTPUT_FILENAME = "Pluck5.wav" #edit filename as appropriate

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                input_device_index = 1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("done recording")

#stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE*0.001)
wf.writeframes(b''.join(frames))
wf.close()
