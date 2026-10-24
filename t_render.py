"""Render one frame of terrain to a PNG so the pipeline can be eyeballed."""
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/user/airliner-sim")
from sim.mathx import euler_to_quat, dcm, DEG
from sim.renderer import Renderer
from sim.terrain import Terrain

W, H = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (480, 270)
pos = np.array((float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))) if len(sys.argv) > 5 \
    else np.array((-1500.0, 9000.0, -850.0))
hdg = float(sys.argv[6]) if len(sys.argv) > 6 else 288.0
pit = float(sys.argv[7]) if len(sys.argv) > 7 else -6.0
sun = float(sys.argv[8]) if len(sys.argv) > 8 else 38.0

t0 = time.time()
ter = Terrain()
print("terrain %.2fs" % (time.time() - t0))

r = Renderer(W, H, hfov=70.0)
r.sky.set(sun, 112.0)
r.fog_range = 22000.0
q = euler_to_quat(0.0, pit * DEG, hdg * DEG)
R = dcm(q)
r.set_camera(pos, R)

times = {}
N = 12
for i in range(N):
    t0 = time.time()
    r.begin()
    times.setdefault("sky", []).append(time.time() - t0)
    t0 = time.time()
    agl = -pos[2] - ter.height(pos[0], pos[1])
    bl = ter.base_level(max(agl, 0.0))
    v, t, c, m, b = ter.ring_mesh(pos[0], pos[1], bl, cam_agl=max(agl, 0.0))
    times.setdefault("mesh", []).append(time.time() - t0)
    t0 = time.time()
    r.add_mesh(v, t, c, mat=m, bias=b)
    times.setdefault("shade", []).append(time.time() - t0)
    t0 = time.time()
    fb = r.draw()
    times.setdefault("draw", []).append(time.time() - t0)

print("stats", r.stats, "terrain tris", t.shape[0])
tot = 0.0
for k, vv in times.items():
    ms = float(np.mean(vv[2:])) * 1000
    tot += ms
    print("  %-6s %6.2f ms" % (k, ms))
print("  TOTAL  %6.2f ms  -> %.1f fps" % (tot, 1000.0 / tot))
Image.fromarray(fb).save("/home/user/airliner-sim/tools/out.png")
print("saved")
