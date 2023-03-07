import socket 
import signal
# import sys
import pyaudio
# import wave

continue_recording = True 

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    global continue_recording
    continue_recording = False
    # sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
# print('Press Ctrl+C')
# signal.pause()

def custom_connect(addr = 'localhost', port = 50001):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((addr, port))
    print('Connected?')
    return s

def main(login, port = 5007, size = 16000):
    global continue_recording

    CHUNK = 256
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    # RECORD_SECONDS = 5
    # WAVE_OUTPUT_FILENAME = "output.wav"

    sock = None
    try:
        sock = custom_connect(port = port)
    except:
        sock = None

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    print("* recording")

    frames = []

    # for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    chunk_counter = 0
    while continue_recording:
        data = stream.read(CHUNK)
        # print(chunk_counter)
        chunk_counter += 1
        # frames.append(data)
        sock.sendall(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':
    main('penguin-1', port = 50002)