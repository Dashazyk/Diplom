#!/usr/bin/env python3

import math
import sys
import cv2
import numpy as np
import json
import threading
from flask import Flask, request
import pandas

from utils.camera_utils import Camera, Vector3, Triag, mult, ray

from deepface import DeepFace
from flask.logging import default_handler


class Server:
    # all_client_data = {}
    # camera: Camera = Camera(Vector3(8.0, 5.0, -3), 0.00, 0.00, 800, 600, 55)
    def __init__(self, cameras: list) -> None:
        self.all_obj_data: list = []
        self.faced_ids: list = []
        
        for cam in cameras:
            print(cam)
            self.all_obj_data.append([])
            self.faced_ids.append({})

        uthread = threading.Thread(target = self.upload_data, args = ())
        uthread.setDaemon(True)
        uthread.start()
        self.observer = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'id': '-1'
        }
        # TODO: proper Camera init later
        self.cameras: list = cameras.copy()
        self.cdelta: Camera = Camera(Vector3(0.1, 0.2,  0), 0.05, 0.00)

        points: list = [
            Vector3(-900,  900, 0),
            Vector3( 900, -900, 0),
            Vector3(-900, -900, 0),
            Vector3( 900,  900, 0)
        ]
        self.floor: list = [
            Triag(points[1], points[0], points[2]),
            Triag(points[0], points[1], points[3])
        ]
        # ID людей. если на очередном шаге ИД человек пришёл тот, 
        # которого раньше не было, надо его засунуть в ГлубокоеЛицо 
        # с целью выяснения, а чьё оно, это лицо,
        # и запомнить соответствие ГлубокоПотоковому ID


        # x = threading.Thread(target=SoundServer, args=('localhost',))
        # sound_server = SoundServer()
        # sound_server
        # x.start()
        
    def upload_data(self):
        api = Flask(__name__)
        api.logger.removeHandler(default_handler)

        @api.route('/people', methods=['GET'])
        def get_people():
            full_list = []
            for cam_batch in self.all_obj_data:
                full_list.extend(cam_batch)
            full_list.append( self.observer )
            return json.dumps(full_list)
        
        @api.route('/cameras', methods=['GET'])
        def get_cameras():
            cam_list = [camera.dict() for camera in self.cameras]
            return json.dumps(cam_list)
        
        @api.route('/observer', methods=['PATCH'])
        def move_observer():
            data = request.json
            if 'dx' in data:
                self.observer['x'] += data['dx']
            if 'dy' in data:
                self.observer['y'] += data['dy']
            if 'dz' in data:
                self.observer['z'] += data['dz']

            return json.dumps(self.observer)

        api.run(host='0.0.0.0')

    def calc_object_positions(self, cam_idx, boxes):
        camera = self.cameras[cam_idx]
        # print('  camera:', camera)
        z: float = 1.0 / math.tan(camera.fov * math.pi / 360.0)

        dw = 2.0 / (camera.width  - 1.0)
        dh = 2.0 / (camera.height - 1.0)

        # ct = camera.clone()
        # ct.position = ct.position.add(self.cdelta.position.scale(frame_no))
        # pc = ct.position
        # ct.h_rotation += self.cdelta.h_rotation * frame_no
        # ct.v_rotation += self.cdelta.v_rotation * frame_no
        # pv = ct.get_view_destination()
        pc = camera.position
        pv = camera.get_view_destination()

        print(f'  pc: {pc}')
        print(f'  pv: {pv}')

        bz = pv.diff(pc).norm()
        bx = bz.prod(Vector3(0, 0, 1)).norm()
        by = bx.prod(bz).norm()

        # print('pc: ', pc)
        # print('pv: ', pv)

        points = []

        for box in boxes:
            tx: float = box.x
            ty: float = box.y

            tx = -1.0 + dw * tx
            ty = -1.0 + dh * ty            
            
            v = Vector3(
                tx, 
                ty * camera.height / camera.width, 
                z
            )

            dir_seen = mult(bx, by, bz, v).norm()
            point = ray(self.floor, camera.position, dir_seen)
            print('  point: {point}')
            points.append( point )

        return points

    def generate_image(self, step, frame_no = 0, frames = 1):
        # image = None
        camera = self.cameras[0]
        image = np.zeros((camera.height, camera.width, 3), np.uint8)
        z: float = 1.0 / math.tan(camera.fov * math.pi / 360.0);

        dw = 2.0 / (camera.width  - 1.0)
        dh = 2.0 / (camera.height - 1.0)

        ct = camera.clone()
        ct.position = ct.position.add(self.cdelta.position.scale(frame_no))
        pc = ct.position
        ct.h_rotation += self.cdelta.h_rotation * frame_no
        ct.v_rotation += self.cdelta.v_rotation * frame_no
        pv = ct.get_view_destination()
        

        bz = pv.diff(pc).norm()
        bx = bz.prod(Vector3(0, 0, 1)).norm()
        by = bx.prod(bz).norm()

        for y in range(0, camera.height, step):
            for x in range(0, camera.width, step):
                tx = -1.0 + dw * x
                ty = -1.0 + dh * y            
                
                v = Vector3(
                    tx, 
                    ty * camera.height / camera.width, 
                    z
                )

                dir_seen = mult(bx, by, bz, v).norm()
                point = ray(self.floor, pc, dir_seen)
                if point:
                    cr = 70 if math.fabs(point.x)//1 % 2 == 0 else 0
                    cg = 70 if math.fabs(point.y)//1 % 2 == 0 else 0
                    # cb = 127 if math.fabs(point.z)//1 % 2 == 0 else 0
                    cb = 80 if point.x * point.y > 0 else 30

                    image[y:y+step,x:x+step] = [cb, cg, cr]

        return image

    def add_new_faces(self, cam_idx, ids, faces_path, db_path = '/home/dasha/Pictures/Faces'):
        new_ids = set(ids) - set(self.faced_ids[cam_idx].keys())
        if new_ids:
            print(new_ids)
            # идов пришло больше чем знаем
            # print(new_ids)
            for id in new_ids:
                # print(id)
                # self.faced_ids[id] = None
                try:
                    found_face = DeepFace.find(
                        # TODO
                        img_path=f'{faces_path}/face_{id}.jpg',
                        db_path = '/home/dasha/Pictures/face_db/',
                        model_name = 'SFace',
                        detector_backend = 'dlib'
                    )
                    # self.faced_ids[cam_idx][id] = found_face
                    self.faced_ids[cam_idx][id] = (found_face)[0]['identity'].iloc[0].split('/')[-2]
                    print('/================\\')
                    print(f'{id}:', self.faced_ids[cam_idx][id])
                    print('\\================/')
                except Exception as e:
                    print('Exception:', e)

    def add_people(self, cam_idx, boxes: list, ids: list = None) -> list:
        # for box in boxes:
        #     print('    box:', box)

        positions = self.calc_object_positions(cam_idx, boxes)

        # new_ids = set(ids) - set(self.faced_ids.keys())
        # if new_ids:
        #     pass
        #     # идов пришло больше чем знаем
        #     # print(new_ids)
        #     for id in new_ids:
        #         self.faced_ids[id] = None
        #     #     self.faced_ids[id] = DeepFace.find(
        #     #         # TODO
        #     #         img_path='TODO:',
        #     #         db_path = '/home/dasha/Pictures/Faces',
        #     #         model_name = 'SFace',
        #     #         detector_backend = 'dlib'
        #     #     )

        pd = []

        for pidx, position in enumerate(positions):
            print('    position:', pidx, position)
            if position:
                j_dict = position.dict().copy()
                if ids:
                    id = ids[pidx]
                    j_dict['id'] = id
                    if id in self.faced_ids[cam_idx]:
                        face_name = self.faced_ids[cam_idx][id]
                        print('face_name:', face_name)
                        # if face_name and (isinstance(face_name, pandas.DataFrame) and not face_name.empty):
                        j_dict['face'] = self.faced_ids[cam_idx][id]
                pd.append(j_dict)

        # self.all_obj_data = json.dumps(pd, indent = 4)
        # pd.append(self.observer)
        self.all_obj_data[cam_idx] = pd

        return positions

    def test_run(self, frames, step):
        boxes = [
            Vector3(100, 200, 0),
            Vector3(700, 500, 0),
            Vector3(400, 400, 0)
        ]
        ids = [5, 9, 11]
        points = self.run(boxes, ids)
        image = self.generate_image(step)
        for box in boxes:
            image[box.y-2:box.y+2,box.x-2:box.x+2] = [0, 0, 0]
            image[box.y-1:box.y+1,box.x-1:box.x+1] = [255, 255, 255]
        cv2.imwrite(f'images/w_boxes.jpg', image)
        sys.stdout.flush()

        for k in range(frames):
            print(f'Frame #{k}')
            image = self.generate_image(step, k, frames)
            cv2.imwrite(f'images/{k}.jpg', image)
            sys.stdout.flush()



if __name__ == '__main__':
    serv = Server()
    serv.test_run(15, 4)
