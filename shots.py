"""Render a set of viewpoints to PNGs for visual checks."""
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/user/airliner-sim")
from sim.mathx import euler_to_quat, dcm, DEG
from sim.renderer import Renderer
from sim.terrain import Terrain

SHOTS = {
    "coast":   ((-2000.0, 22000.0, -2600.0), 285.0, -7.0, 38.0),
    "mountain": ((8000.0, -14000.0, -2200.0), 18.0, -4.0, 22.0),
    "city":    ((-6000.0, 2000.0, -900.0), 214.0, -7.0, 55.0),
    "final":   ((-60.0, 12500.0, -420.0), 270.0, -3.2, 30.0),
    "high":    ((-10000.0, 12000.0, -9500.0), 300.0, -10.0, 20.0),
}

name = sys.argv[1] if len(sys.argv) > 1 else "coast"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 480
H = int(sys.argv[3]) if len(sys.argv) > 3 else 270
pos, hdg, pit, sun = SHOTS[name]
pos = np.array(pos)

ter = Terrain()
r = Renderer(W, H, hfov=70.0)
r.sky.set(sun, 112.0)
r.fog_range = 26000.0
r.set_camera(pos, dcm(euler_to_quat(0.0, pit * DEG, hdg * DEG)))
t0 = time.time()
r.begin()
agl = -pos[2] - ter.height(pos[0], pos[1])
rings = ter.lod_plan(max(agl, 0.0), pit)
v, t, c, m, b = ter.ring_mesh(pos[0], pos[1], 0, rings=rings)
r.add_mesh(v, t, c, mat=m, bias=b)
fb = r.draw()
print("%s: %d tris, %.1f ms total" % (name, r.stats["drawn"], (time.time() - t0) * 1000))
Image.fromarray(fb).save("/home/user/airliner-sim/tools/out_%s.png" % name)
