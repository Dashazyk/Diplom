import math

class Vector3:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def copy(self):
        return Vector3(
            self.x,
            self.y,
            self.z
        )

    @classmethod
    def from_tuple(cls, _tuple):
        return cls(
            _tuple[0] if len(_tuple) > 0 else 0,
            _tuple[1] if len(_tuple) > 1 else 0,
            _tuple[2] if len(_tuple) > 2 else 0
        )
    
    @classmethod
    def from_json(cls, jtxt):
        return cls(
            jtxt['x'], jtxt['y'], jtxt['y']
        )

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

    def dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'z': self.z
        }


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
    
    def dict(self):
        r = self.__dict__.copy()
        r['position'] = r['position'].dict()
        return r

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