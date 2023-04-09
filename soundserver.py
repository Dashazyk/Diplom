#!/usr/bin/env python3

import threading
import json
import socket
import wave
import pyaudio
import numpy as np
import requests

import signal
import sys

from time import sleep
from camera_utils import Vector3
from pydub import AudioSegment
from pydub.playback import play


# def signal_handler(sig, frame):
#     print('You pressed Ctrl+C!')
#     sys.exit(0)

# print('Press Ctrl+C')
# signal.pause()

class SoundServer:
    def __init__(self, api_params={'host': 'localhost', 'port': 5000}, self_params={'host': 'localhost', 'port': 50002}) -> None:
        self.self_params: dict = self_params
        self.api_params:  dict = api_params
        self.sound_chunk_queue: list = []
        self.chunk_size: int = 256
        self.running: bool = True
        self.can_play: bool = False

        self.observer_pos = Vector3(0, 0, 0)
        self.sounndev_pos = Vector3(0, 0, 0)
        # self.socket = None

    def stop_signal_handler(self, sig, frame):
        self.running = False
        print('\nReceived interrrupt signal')
        # if self.socket:
        #     self.socket.close()

        # to force the socket.listen.close()
        socket.socket(
            socket.AF_INET, 
            socket.SOCK_STREAM
        ).connect( (self.self_params['host'], self.self_params['port']))
        
    def observer_loop(self):
        while self.running:
            try:
                result = json.loads(requests.get(f"http://{self.api_params['host']}:{self.api_params['port']}/people").text)
                obs = {}
                for obj in result[::-1]:
                    if obj['id'] == '-1':
                        obs = obj
                        break

                self.observer_pos = Vector3(
                    obs['x'],
                    obs['y'],
                    obs['z']
                )
                # print(self.observer_pos)
            except Exception as e:
                print(e)
                pass

            sleep(0.1)

    def receive_loop(self):
        self_host = self.self_params['host']  # Standard loopback interface address (localhost)
        self_port = self.self_params['port']  # Port to listen on (non-privileged ports are > 1023)

        while self.running:
            print('Opening socket')
            num = 0
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    # self.socket = s
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("0.0.0.0", self_port))
                    print('binded')
                    s.listen()
                    print('listened')
                    conn, addr = s.accept()
                    print('accepted')
                    print('Opened socket')
                    with conn:
                        print(f"Connected by {addr}")
                        while self.running:
                            try:
                                data = conn.recv(self.chunk_size)
                            except:
                                print('hmm has the client died or something?')
                            if not data:
                                break
                            # conn.sendall(data)
                            # print(num, data[0])
                            # frames.append(data)

                            # stream.write(data)
                            self.sound_chunk_queue.append(data)

                            # for i in xrange(0, len(signal), lframe):
                            # data = np.frombuffer(data)
                            # sd.play(data, RATE, blocking=False)
                            num += 1 

                            if len(self.sound_chunk_queue) > 30:
                                self.can_play = True
            except Exception as e:
                print(e)

    def playback_loop(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        
        observer_distance = 0

        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=self.chunk_size
        )

        while self.running:
            if self.observer_pos:
                observer_distance = self.observer_pos.diff(self.sounndev_pos).len()

            if self.can_play:
                chunks = self.sound_chunk_queue[0:4]
                chunk = b''.join(chunks)
                # chunk = self.sound_chunk_queue[0]
                audio_segment = AudioSegment(
                    chunk, 
                    frame_rate=RATE,
                    sample_width=2, 
                    channels=CHANNELS
                )
                # print(observer_distance)
                if observer_distance:
                    audio_segment -= observer_distance
                stream.write(audio_segment.raw_data)
                # play(audio_segment)

                # self.sound_chunk_queue.pop(0)
                # self.sound_chunk_queue.pop(0)
                # self.sound_chunk_queue.pop(0)
                # self.sound_chunk_queue.pop(0)
                del self.sound_chunk_queue[0:4]
                # del self.sound_chunk_queue[0]
                # del self.sound_chunk_queue[0]
                # del self.sound_chunk_queue[0]

                if len(self.sound_chunk_queue) < 5:
                    self.can_play = False

        # after we've stopped
        # for chunk in self.sound_chunk_queue:
        #     stream.write(chunk)

    def run(self):
        thread_functions = [
            self.receive_loop,
            self.playback_loop,
            self.observer_loop,
        ]
        # threads = []
        for thread_function in thread_functions:
            thread = threading.Thread(target=thread_function)
            thread.start()


        # receive.start()
        # playbck.start()


if __name__ == '__main__':
    server = SoundServer()
    signal.signal(signal.SIGINT, server.stop_signal_handler)
    server.run()
