import socket 
import signal
# import sys
import pyaudio
# import wave

continue_recording = True 
    # sys.exit(0)
# print('Press Ctrl+C')
# signal.pause()

class MicroRecorder:
    def signal_handler(self, sig, frame):
        print('Termination signal received')
        # global continue_recording
        self.running = False

    def __init__(self, addr, port, size = 16000):
        self.addr = addr
        self.port = port
        self.size = size
        self.running = True

    def record_sound(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        CHUNK = 256
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        # RECORD_SECONDS = 5
        # WAVE_OUTPUT_FILENAME = "output.wav"

        print(self.addr, self.port)

        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.addr, self.port))
        except Exception as e:
            sock = None
            print('ERROR:', e)

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
        while self.running:
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
    # signal.signal(signal.SIGINT, signal_handler)
    micro_recorder = MicroRecorder('localhost', port = 50002)
    micro_recorder.record_sound()