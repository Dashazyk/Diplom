#!/usr/bin/env python3

import sys
import pygame
import requests
import threading
import time
import json
from utils.camera_utils import Vector3
from micro_recorder import MicroRecorder
from multiprocessing import Process
import signal
import math as m

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_a,
    K_w,
    K_d,
    K_s,
    K_PLUS,
    K_EQUALS,
    K_MINUS,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)

class Tester:
    def __init__(self, url) -> None:
        self.font    = None
        self.url     = url
        self.running = False
        self.people  = {}
        self.cameras = {}
        # self.screen  = None
        self.local_position = Vector3(0, 0, 0)
        self.scale = 30
        self.finished_drawing: bool = True

    def change_observer_position(self, dx, dy):
        url = self.url + '/observer'
        data = {
            'dx': dx,
            'dy': dy
        }
        requests.patch(url, json=data)

    def get_people(self, timeout = 100):
        cold_time = 0
        pold_time = 0
        new_time  = 0
        urlp = self.url + '/people'
        urlc = self.url + '/cameras'

        was_ok = True
        while self.running:
            new_time = time.time()
            if (new_time - pold_time)*1000 > (timeout if was_ok else timeout * 10):
                try:
                    pold_time = new_time
                    response = requests.get(urlp)
                    # print(people)
                    self.people = json.loads(response.text)
                    was_ok = True
                except Exception as e:
                    was_ok = False
                    print(e)

            if (new_time - cold_time)*1000 > (timeout*10 if was_ok else timeout * 10):
                # print('hmm')
                try:
                    cold_time = new_time
                    response = requests.get(urlc)
                    # print(response.text)
                    # print(people)
                    self.cameras = json.loads(response.text)
                    was_ok = True
                except Exception as e:
                    was_ok = False
                    print(e)

    def draw_background(self, screen):
        w, h = screen.get_size()
        scale  = self.scale
        l_pos  = self.local_position.copy()
        dw = l_pos.x % scale
        dh = l_pos.y % scale
        for x in range(dw, w+dw, scale):
            pygame.draw.line(screen, (50, 50, 50), (x, 0), (x, h))
        for y in range(dh, h+dh, scale):
            pygame.draw.line(screen, (50, 50, 50), (0, y), (w, y))

        # pygame.draw.circle(
        #     screen, 
        #     (255, 0, 0),
        #     [l_pos.x, l_pos.y], 
        #     int(scale * 1.75), 
        #     0
        # )
        pygame.draw.rect(screen, ((50, 50, 50)), pygame.Rect(0,         l_pos.y-2, w, 4))
        pygame.draw.rect(screen, ((50, 50, 50)), pygame.Rect(l_pos.x-2, 0,         4, h))

    def draw_people(self, screen):
        people = self.people
        scale  = self.scale
        l_pos  = self.local_position.copy()
        sw, sh = screen.get_size()
        for person in people:
            ppos = Vector3(
                person['x'],
                person['y'],
                0
            )
            person_color = (127, 127, 127)
            if int(person["id"]) == -1:
                person_color = (0, 0, 127)
            ppos = ppos.scale(scale)
            ppos = ppos.add  (l_pos)
            pygame.draw.circle(
                screen, 
                person_color,
                (int(ppos.x), int(ppos.y)), 
                int(scale * 0.5), 
                0
            )

            # print(person)
            # img = None
            person_text = None
            person_text_color = (255, 70, 255)

            if int(person['id']) >= 0:
                person_text = person.get("face", person['id'])
            else:
                person_text = 'You'
                # person_text_color = (255, 70, 255)
            # if 'face' in person:
            #     img = self.font.render(f'{person["face"]}', True, (255, 70, 255))
            # elif 'id' in person:
            #     img = self.font.render(f'{person["id"]}', True, (255, 70, 255))

            if person_text:
                img = self.font.render(f'{person_text}', True, person_text_color)
                img = pygame.transform.flip(img, False, True)
                iw, ih = img.get_size()
                # ppos = ppos.add  (Vector3(0, -scale, 0))
                ppos = ppos.add(Vector3(-iw / 2, -ih / 2, 0))
                screen.blit(img, (int(ppos.x), int(ppos.y)))
                # screen.blit(img, (50, 50))
            # else:
                

    def draw_camera(self, screen):
        w, h = screen.get_size()
        scale  = self.scale
        l_pos  = self.local_position.copy()
        if self.cameras:
            for camera in self.cameras:
                cpos = Vector3.from_json(camera['position'])
                cpos.z = 0
                cpos = cpos.scale(scale)
                cpos = cpos.add  (l_pos)

                if cpos.x > -scale and cpos.y > -scale and cpos.x < w + scale and cpos.y < h + scale:
                    hangle = float(camera['h_rotation'])
                    dir_vector = Vector3(
                        scale * m.sin(hangle),
                        scale * m.cos(hangle),
                        0
                    )
                    dir_end = cpos.add(dir_vector)

                    pygame.draw.line(
                        screen, 
                        ((50, 127, 50)), 
                        [int(cpos.x),    int(cpos.y)   ], 
                        [int(dir_end.x), int(dir_end.y)]
                    )
                    pygame.draw.circle(
                        screen, 
                        ((0, 0, 0)),
                        [int(cpos.x), int(cpos.y)], 
                        int(scale * 0.3), 
                        0
                    )
                    pygame.draw.circle(
                        screen, 
                        ((50, 127, 50)),
                        [int(cpos.x), int(cpos.y)], 
                        int(scale * 0.3), 
                        3
                    )

    def draw_mousepos(self, screen):
        sw, sh = screen.get_size()
        cx, cy = pygame.mouse.get_pos()
        l_pos  = self.local_position.copy()
        scale  = self.scale
        # pygame.draw.circle(
        #     screen, 
        #     ((100, 0, 0)),
        #     [int(x), int(h - y)], 
        #     int(self.scale * 1), 
        #     0
        # )
        rx = cx - l_pos.x
        ry = sh - cy - l_pos.y
        rx /= scale
        ry /= scale
        rx = round(rx, 1)
        ry = round(ry, 1)
        img = self.font.render(f'{rx}, {ry}', True, (25, 255, 255))
        img = pygame.transform.flip(img, False, True)
        iw, ih = img.get_size()
        screen.blit(img, (int(cx - iw/2), int(sh - cy)))

    def draw_screen(self, screen):
        self.finished_drawing = False
        screen.fill((0, 0, 0))
        self.draw_background(screen)
        self.draw_people    (screen)
        self.draw_camera    (screen)
        self.draw_mousepos  (screen)

        self.finished_drawing = True

    def main(self):
        # url = 'http://localhost:5000/'

        # Microphone-recording code needs to be above or pygame will capture the micro and nothing will work
        pygame.init()
        self.font = pygame.font.SysFont('monospace', 18, bold=True)

        # Set up the drawing window
        def_w = 800
        def_h = 800
        real_screen = pygame.display.set_mode((def_w, def_h), pygame.RESIZABLE)
        canvas      = pygame.Surface((def_w, def_h))
        
        self.local_position = Vector3(def_w//2, def_h//2, 0)

        # Run until the user asks to quit
        self.running = True

        api_thread = threading.Thread(target = self.get_people)
        api_thread.start()

        prev_mpos: Vector3 = Vector3(0, 0, 0)
        mouse_tracking: bool = False
        prev_time = time.time() * 1000

        # cw, ch  = canvas.get_size()
        
        while self.running:
            new_time = time.time() * 1000
            current_mpos = None

            for event in pygame.event.get():

                # print("What is this event? ", event)

                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if   event.key == K_ESCAPE:
                        self.running = False

                    if   event.key == K_w:
                        self.change_observer_position( 0,  1)
                    elif event.key == K_s:
                        self.change_observer_position( 0, -1)
                    elif event.key == K_d:
                        self.change_observer_position( 1,  0)
                    elif event.key == K_a:
                        self.change_observer_position(-1,  0)

                    # if event.key == K_UP:
                    #     self.local_position = self.local_position.add(Vector3(0, 3, 0))
                    # elif event.key == K_DOWN:
                    #     self.local_position = self.local_position.add(Vector3(0, -3, 0))
                    # elif event.key == K_LEFT:
                    #     self.local_position = self.local_position.add(Vector3(-3, 0, 0))
                    # elif event.key == K_RIGHT:
                    #     self.local_position = self.local_position.add(Vector3(3, 0, 0))

                    if event.key in [K_EQUALS, K_PLUS]:
                        self.scale += 5
                    elif event.key == K_MINUS:
                        self.scale -= 5
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_tracking = True
                    prev_mpos = Vector3.from_tuple(pygame.mouse.get_pos())

                    if event.button == 5:
                        self.scale -= 5
                    elif event.button == 4:
                        self.scale += 5
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_tracking = False
                elif event.type == pygame.MOUSEMOTION:
                    if mouse_tracking:
                        raw_mpos = pygame.mouse.get_pos()
                        if raw_mpos and len(raw_mpos) == 2:
                            current_mpos: Vector3 = Vector3.from_tuple(raw_mpos)
                        
                            mouse_diff: Vector3 = current_mpos.diff(prev_mpos)
                            mouse_diff.y = -mouse_diff.y
                            self.local_position = self.local_position.add(mouse_diff)
                        prev_mpos = current_mpos
                elif event.type == pygame.VIDEORESIZE:
                    cw, ch  = canvas.get_size()
                    self.local_position.y += event.size[1] - ch
                    canvas = pygame.Surface(event.size)
                
                if self.scale < 15:
                    self.scale = 15

            # limit to 720 fps (hopefully)
            if (new_time - prev_time) >= 1000 / 720 and self.finished_drawing:
                    # let's apply the results of previous render
                    # rw, rh = real_screen.get_size()
                    # if rw == cw and rh == ch:
                    canvas = pygame.transform.flip(canvas, False, True)
                    real_screen.blit(canvas, (0, 0))
                    pygame.display.update()

                    # canvas = pygame.Surface(real_screen.get_size())
                    # cw, ch  = canvas.get_size()
                    draw_thread = threading.Thread(target = self.draw_screen, args = (canvas,))
                    draw_thread.start()

        micro_process.terminate()


        # Done! Time to quit.
        pygame.quit()


if __name__ == '__main__':
    url = 'http://localhost:5000/'
    if len(sys.argv) > 1:
        url = sys.argv[1]
    tester = Tester(url)

    micro_recorder = MicroRecorder(url.split(':')[1].lstrip('/'), 50002)
    micro_process  = Process(target=micro_recorder.record_sound)
    micro_process.start()

    tester.main()