# Simple pygame program

# Import and initialize the pygame library
import pygame
import requests
import threading
import time
import json
from camera_utils import Vector3

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
        self.url = url
        self.running = False
        self.people = {}
        self.screen = None
        self.local_position = Vector3(0, 0, 0)
        self.scale = 30

    def change_observer_position(self, dx, dy):
        url = self.url + '/observer'
        data = {
            'dx': dx,
            'dy': dy
        }
        requests.post(url, json=data)

    def get_people(self, timeout = 100):
        old_time = 0
        new_time = 0
        while self.running:
            new_time = time.time()
            if (new_time - old_time)*1000 > timeout:
                url = self.url + '/people'
                response = requests.get(url)
                # print(people)
                self.people = json.loads(response.text)
                old_time = new_time
            # sleep(1)


    def draw_people(self):
        people = self.people
        # screen = self.screen
        scale  = self.scale
        l_pos  = self.local_position

        w, h  = self.screen.get_size()
        # screen = pygame

        screen = pygame.Surface((w, h))
        screen.fill((0, 0, 0))

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
        pygame.draw.rect(screen, ((50, 50, 50)), pygame.Rect(0, l_pos.y-2, w, 4))
        pygame.draw.rect(screen, ((50, 50, 50)), pygame.Rect(l_pos.x-2, 0, 4, h))

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
            ppos = ppos.add(l_pos)
            pygame.draw.circle(
                screen, 
                person_color,
                [int(ppos.x), int(ppos.y)], 
                int(scale * 0.5), 
                0
            )
        screen = pygame.transform.flip(screen, False, True)
        self.screen.blit(screen, (0, 0))


    def main(self):
        # url = 'http://localhost:5000/'
        pygame.init()
        # Set up the drawing window
        self.screen = pygame.display.set_mode([700, 700], pygame.RESIZABLE)
        self.local_position = Vector3(700//2, 700//2, 0)

        # Run until the user asks to quit
        self.running = True

        api_thread = threading.Thread(target = self.get_people)
        api_thread.start()
        prev_mpos: Vector3 = Vector3(0, 0, 0)
        mouse_tracking: bool = False
        while self.running:
            current_mpos = None
            # try:
                
            #     print(current_mpos)
            # except: 
            #     pass

            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False
                elif event.type == KEYDOWN:
                    if   event.key == K_ESCAPE:
                        self.running = False

                    if   event.key == K_w:
                        self.change_observer_position(0,  1)
                    elif event.key == K_s:
                        self.change_observer_position(0, -1)
                    elif event.key == K_d:
                        self.change_observer_position(1,  0)
                    elif event.key == K_a:
                        self.change_observer_position(-1,  0)

                    if event.key == K_UP:
                        self.local_position = self.local_position.add(Vector3(0, 3, 0))
                    elif event.key == K_DOWN:
                        self.local_position = self.local_position.add(Vector3(0, -3, 0))
                    elif event.key == K_LEFT:
                        self.local_position = self.local_position.add(Vector3(-3, 0, 0))
                    elif event.key == K_RIGHT:
                        self.local_position = self.local_position.add(Vector3(3, 0, 0))

                    if event.key in [K_EQUALS, K_PLUS]:
                        self.scale += 5
                    elif event.key == K_MINUS:
                        self.scale -= 5
                        if self.scale < 5:
                            self.scale = 5
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_tracking = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_tracking = False
                elif event.type == pygame.MOUSEMOTION:
                    raw_mpos = pygame.mouse.get_pos()
                    # print(raw_mpos)
                    if raw_mpos and len(raw_mpos) == 2:
                        current_mpos: Vector3 = Vector3.from_tuple(raw_mpos)
                        if mouse_tracking:
                            mouse_diff: Vector3 = current_mpos.diff(prev_mpos)
                            mouse_diff.y = -mouse_diff.y
                            self.local_position = self.local_position.add(mouse_diff)
                        prev_mpos = current_mpos
                elif event.type == pygame.VIDEORESIZE:
                    # There's some code to add back window content here.
                    self.screen = pygame.display.set_mode(
                        (event.w, event.h),
                        pygame.RESIZABLE
                    )

            draw_thread = threading.Thread(target = self.draw_people)
            draw_thread.start()
            # self.draw_people()
            # Fill the background with white
            # self.screen.fill((255, 255, 255))

            # Draw a solid blue circle in the center
            # pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)

            # Flip the display
            # pygame.display.flip()
            pygame.display.update()


        # Done! Time to quit.
        pygame.quit()


if __name__ == '__main__':
    tester = Tester('http://localhost:5000/')
    tester.main()