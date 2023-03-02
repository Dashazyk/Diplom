import math
import sys
import cv2
import numpy as np
import json
import threading
from flask import Flask

class Vector3:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def dict(self):
        return {
            'x': round(self.x, 5),
            'y': round(self.y, 5),
            'z': round(self.z, 5),
        }

    def __repr__(self):
        return f'({round(self.x, 3)}, {round(self.y, 3)}, {round(self.z, 3)})'

    def clone(self):
        return Vector3(self.x, self.y, self.z)

    def add(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def len(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def norm(self):
        # l = self.len()
        # return Vector3(self.x / l, self.y / l, self.z / l)
        l = math.sqrt(self.dot(self))
        return Vector3(
            self.x / l, 
            self.y / l, 
            self.z / l
        )

    def diff(self, other):
        # print(self.x - other.x)
        # print(self.y - other.y)
        # print(self.z - other.z)
        return Vector3(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )

    def prod(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y, 
            self.z * other.x - self.x * other.z, 
            self.x * other.y - self.y * other.x
        )

    def dot(self, other) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def scale(self, m: float):
        return Vector3(self.x * m, self.y * m, self.z * m)

class SphericCoords():
    def __init__(self, r, z, phi):
        self.radius  = r
        self.v_angle = z
        self.h_angle = phi

def mult(a: Vector3, b: Vector3, c: Vector3, v: Vector3):
	return Vector3(
		a.x * v.x + b.x * v.y + c.x * v.z,
		a.y * v.x + b.y * v.y + c.y * v.z,
		a.z * v.x + b.z * v.y + c.z * v.z
    )


class Triag:
    def __init__(self, a: Vector3, b: Vector3, c: Vector3) -> None:
        self.a: Vector3 = a
        self.b: Vector3 = b
        self.c: Vector3 = c


class Camera:
    def __init__(self, position: Vector3 = Vector3(0, 0, 0), h_rotation: float = 0, v_rotation: float = 0, width: float = 800, height: float = 600, fov: float = 60):
        self.width:  float = width
        self.height: float = height
        self.fov:    float = fov

        self.v_rotation: float = v_rotation
        self.h_rotation: float = h_rotation

        self.position = position

    def get_view_destination(self):
        view_pos = Vector3(0, 0, 0)
        hrad     = math.cos(self.v_rotation)
        pos      = self.position
        
        view_pos.x = pos.x + math.cos(self.h_rotation) * hrad
        view_pos.y = pos.y + math.sin(self.h_rotation) * hrad
        view_pos.z = pos.z + math.sin(self.v_rotation)

        return view_pos

    def clone(self):
        return Camera(
            self.position.clone(),
            self.width,
            self.height,
            self.fov,
            self.h_rotation,
            self.v_rotation
        )

# def raytrace_pixel(triags, y, x, cam_pos: Camera, bx, by, bz):
#     tx = -1.0 + dw * x
#     ty = -1.0 + dh * y            
    
#     v = Vector3(
#         tx, 
#         ty * h / w, 
#         z
#     )

#     dir_seen = mult(bx, by, bz, v).norm()
#     point = ray(triags, cam_pos, dir_seen)


def ray(triags: list, pos: Vector3, dir: Vector3): 
    ts_min: int = None

    dir = dir.norm()

    for tidx, triag in enumerate(triags):
        side1: Vector3 = triag.b.diff(triag.a)
        side2: Vector3 = triag.c.diff(triag.a)

        p   = dir.prod(side2)
        div = p.dot(side1)

        if (math.fabs(div) >= 1e-5):
            t = pos.diff(triag.a)
            u = p.dot(t) / div

            if (not (u < 0.0 or u > 1.0)):
                q = t.prod(side1)
                v = q.dot(dir) / div
                
                if (not (v < 0.0 or (v + u) > 1.0)):
                    ts = q.dot(side2) / div
                    if (ts >= 0.0):
                        if (not ts_min or ts < ts_min):
                            ts_min = ts

    intersection = None
    if ts_min:
        # print('pos', pos.add(dir.scale(-ts_min)))
        intersection: Vector3 = pos.add(dir.scale(ts_min))

    return intersection

from deepface import DeepFace
class Server:
    # all_client_data = {}
    def __init__(self, camera: Camera = Camera(Vector3(8.0, 5.0, -3), 0.00, 0.00, 800, 600, 55)) -> None:
        self.all_obj_data: dict = {}
        uthread = threading.Thread(target = self.upload_data, args = ())
        uthread.setDaemon(True)
        uthread.start()
        self.observer = {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0
        }
        # TODO: proper Camera init later
        self.camera: Camera = camera
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
        self.faced_ids: dict = {}
        
    def upload_data(self):
        api = Flask(__name__)

        @api.route('/people', methods=['GET'])
        def get_companies():
            return json.dumps(self.all_obj_data)
        
        @api.route('/observer', methods=['POST'])
        def move_observer():
            pass

        api.run()

    def calc_object_positions(self, boxes):
        z: float = 1.0 / math.tan(self.camera.fov * math.pi / 360.0);

        dw = 2.0 / (self.camera.width  - 1.0)
        dh = 2.0 / (self.camera.height - 1.0)

        # ct = self.camera.clone()
        # ct.position = ct.position.add(self.cdelta.position.scale(frame_no))
        # pc = ct.position
        # ct.h_rotation += self.cdelta.h_rotation * frame_no
        # ct.v_rotation += self.cdelta.v_rotation * frame_no
        # pv = ct.get_view_destination()
        pc = self.camera.position
        pv = self.camera.get_view_destination()
        

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
                ty * self.camera.height / self.camera.width, 
                z
            )

            dir_seen = mult(bx, by, bz, v).norm()
            point = ray(self.floor, self.camera.position, dir_seen)
            points.append( point )

        return points

    def generate_image(self, step, frame_no = 0, frames = 1):
        # image = None
        image = np.zeros((self.camera.height, self.camera.width, 3), np.uint8)
        z: float = 1.0 / math.tan(self.camera.fov * math.pi / 360.0);

        dw = 2.0 / (self.camera.width  - 1.0)
        dh = 2.0 / (self.camera.height - 1.0)

        ct = self.camera.clone()
        ct.position = ct.position.add(self.cdelta.position.scale(frame_no))
        pc = ct.position
        ct.h_rotation += self.cdelta.h_rotation * frame_no
        ct.v_rotation += self.cdelta.v_rotation * frame_no
        pv = ct.get_view_destination()
        

        bz = pv.diff(pc).norm()
        bx = bz.prod(Vector3(0, 0, 1)).norm()
        by = bx.prod(bz).norm()

        for y in range(0, self.camera.height, step):
            for x in range(0, self.camera.width, step):
                tx = -1.0 + dw * x
                ty = -1.0 + dh * y            
                
                v = Vector3(
                    tx, 
                    ty * self.camera.height / self.camera.width, 
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

    def run(self, boxes: list, ids: list = None) -> list:
        ps = self.calc_object_positions(boxes)

        new_ids = set(ids) - set(self.faced_ids.keys())
        if new_ids:
            # идов пришло больше чем знаем
            print(new_ids)
            # for id in new_ids:
            #     self.faced_ids[id] = DeepFace.find(
            #         # TODO
            #         img_path='TODO:',
            #         db_path = '/home/dasha/Pictures/Faces',
            #         model_name = 'SFace',
            #         detector_backend = 'dlib'
            #     )

        pd = []
        print('stored ids:', self.faced_ids)
        for pidx, p in enumerate(ps):
            # print(p.__dict__ if p else None)
            if p:
                j_dict = p.dict().copy()
                if ids:
                    j_dict['id'] = ids[pidx]
                pd.append(j_dict)

        # self.all_obj_data = json
        print(self.all_obj_data.dumps(pd, indent = 4))
        print()
        
        return ps

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
