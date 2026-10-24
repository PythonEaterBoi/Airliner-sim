"""Headless integration test: fly the ILS with the AP and render frames."""
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/user/airliner-sim")
from sim.engine import Sim

sim = Sim(quality=1.0, w=448, h=252)
ac = sim.ac
ac.reset_situation("approach")
ac.ap.toggle()
ac.ap.arm_app()
ac.ap.toggle_at()
ac.gear_cmd = 1

t = 0.0
shots = (8, 30, 60, 90, 110)
si = 0
while t < 125.0:
    sim.step(1.0 / 24.0)
    t += 1.0 / 24.0
    if si < len(shots) and t >= shots[si]:
        fb = sim.render()
        Image.fromarray(fb).save("/home/user/airliner-sim/tools/fly_%03d.png" % int(t))
        print("t=%5.1f alt=%6.0f ias=%5.1f vs=%6.0f loc=%+.2f gs=%+.2f thr=%3.0f pitch=%+5.1f roll=%+5.1f agl=%6.0f frame=%.1fms"
              % (t, -ac.pos[2] * 3.28084, ac.ias, ac.vs, ac.ap.loc_dev, ac.ap.gs_dev,
                 ac.throttle * 100, ac.pitch, ac.roll, ac.agl * 3.28084, sim.frame_ms))
        si += 1
    if int(t * 10) % 100 == 0 and False:
        pass
print("final: on_ground", ac.on_ground, "pos", ac.pos.round(0), "ias", round(ac.ias, 1),
      "warn", ac.warnings, "events", ac.events[:4])
fb = sim.render()
Image.fromarray(fb).save("/home/user/airliner-sim/tools/fly_end.png")
