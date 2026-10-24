"""Render a full scene (terrain + world + aircraft) from a given pose."""
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/user/airliner-sim")
from sim.mathx import euler_to_quat, dcm, DEG
from sim.renderer import Renderer
from sim.terrain import Terrain
from sim.world import World, _yaw_dcm

SHOTS = {
    "approach": ((0.0, 9000.0, -330.0), 270.0, -3.0, 34.0),
    "rwy": ((-40.0, 900.0, -60.0), 268.0, -2.0, 30.0),
    "chase": ((0.0, 6000.0, -400.0), 270.0, 0.0, 34.0),
    "cityview": ((-7000.0, -2000.0, -500.0), 200.0, -4.0, 40.0),
    "tower": ((-300.0, 520.0, -80.0), 140.0, -6.0, 25.0),
}
name = sys.argv[1] if len(sys.argv) > 1 else "approach"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 480
H = int(sys.argv[3]) if len(sys.argv) > 3 else 270

ter = Terrain()
world = World(ter)
r = Renderer(W, H, hfov=70.0)
r.sky.set(SHOTS[name][3], 112.0)
r.fog_range = 24000.0

if name == "chase":
    ac_pos = np.array((0.0, 6000.0, -400.0))
    R = _yaw_dcm(270.0)
    cam = ac_pos + R @ np.array((-45.0, 0.0, -12.0))
    look = ac_pos - cam
    look = look / np.linalg.norm(look)
    upw = np.array((0.0, 0.0, -1.0))
    right = np.cross(look, upw)
    right /= np.linalg.norm(right)
    upv = np.cross(right, look)
    Rc = np.stack((right, -upv, look), axis=1)  # cols: right, up, fwd -> need body2world
    Rcam = np.stack((right, upv * -1.0, look), axis=1)
    # build DCM with cols (fwd, right, down)
    down = -upv
    Rcam = np.stack((look, right, down), axis=1)
    pos = cam
    pitch = 0.0
else:
    pos = np.array(SHOTS[name][0])
    hdg, pit = SHOTS[name][1], SHOTS[name][2]
    Rcam = dcm(euler_to_quat(0.0, pit * DEG, hdg * DEG))

state = {"clouds": True}
t0 = time.time()
for _ in range(3):
    r.set_camera(pos, Rcam)
    r.begin()
    agl = -pos[2] - ter.height(pos[0], pos[1])
    rings = ter.lod_plan(max(agl, 0.0), -4.0)
    v, t, c, m, b = ter.ring_mesh(pos[0], pos[1], 0, rings=rings)
    r.add_mesh(v, t, c, mat=m, bias=b)
    world.update(0.016)
    world.draw(r, pos, Rcam, state, night=False)
    if name == "chase":
        r.add_mesh(world.ac[0], world.ac[1], world.ac[2], mat=world.ac[3],
                   rot=_yaw_dcm(270.0), pos=ac_pos)
        r.add_mesh(world.gear[0], world.gear[1], world.gear[2], mat=world.gear[3],
                   rot=_yaw_dcm(270.0), pos=ac_pos)
    fb = r.draw()
dt = (time.time() - t0) / 3 * 1000
print("%s: drawn %d tris, %.1f ms" % (name, r.stats["drawn"], dt))
Image.fromarray(fb).save("/home/user/airliner-sim/tools/scene_%s.png" % name)
