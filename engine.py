"""Simulation driver: fixed-step physics, cameras, and frame rendering."""
import time

import numpy as np

from .mathx import DEG, RAD, dcm, quat_to_euler, euler_to_quat, clamp, quat_integrate
from .renderer import Renderer
from .terrain import Terrain
from .world import World, RWY
from .flightmodel import Aircraft
from .meshes import cockpit_mesh

PHYS_DT = 1.0 / 120.0

CAMERAS = ("cockpit", "chase", "wing", "tower")


def _look_dcm(fwd, up=(0.0, 0.0, -1.0)):
    f = np.asarray(fwd, float)
    f = f / max(np.linalg.norm(f), 1e-9)
    u = np.asarray(up, float)
    r = np.cross(f, u)
    n = np.linalg.norm(r)
    if n < 1e-6:
        r = np.array((1.0, 0.0, 0.0))
    else:
        r = r / n
    d = np.cross(f, r)
    return np.stack((f, r, d), axis=1)


class Sim:
    def __init__(self, quality=1.0, w=448, h=252):
        t0 = time.time()
        self.terrain = Terrain()
        self.world = World(self.terrain)
        self.ac = Aircraft(self.terrain)
        self.renderer = Renderer(w, h, hfov=70.0)
        self.quality = quality
        self.camera = "cockpit"
        self.sun_elev = 26.0
        self.sun_azim = 112.0
        self.renderer.sky.set(self.sun_elev, self.sun_azim)
        self.renderer.fog_range = 24000.0
        self.cockpit = cockpit_mesh()
        self._cockpit_key = None
        self._acc = 0.0
        self.chase_pos = None
        self.frame_ms = 0.0
        self.fps = 0.0
        self.time_of_day = 0.62     # 0..1
        self.build_time = time.time() - t0

    # ------------------------------------------------------------- controls
    def set_time(self, tod):
        self.time_of_day = float(np.clip(tod, 0.0, 1.0))
        # map to sun elevation: 0=midnight, .25=sunrise, .5=noon, .75=sunset
        e = np.sin((self.time_of_day - 0.25) * 2 * np.pi) * 62.0
        self.sun_elev = float(e)
        self.renderer.sky.set(self.sun_elev, self.sun_azim)
        self._cockpit_key = None

    def night(self):
        return self.renderer.sky.night > 0.5

    def set_quality(self, q):
        self.quality = float(np.clip(q, 0.4, 1.3))

    def set_resolution(self, w, h):
        self.renderer.resize(w, h)
        self._cockpit_key = None

    # ------------------------------------------------------------------ step
    def step(self, dt):
        n = max(1, int(round(dt / PHYS_DT)))
        h = dt / n
        for _ in range(n):
            self.ac.step(h)
        self.world.update(dt)

    # --------------------------------------------------------------- camera
    def camera_pose(self):
        ac = self.ac
        R = dcm(ac.quat)
        if self.camera == "cockpit":
            pos = ac.pos + R @ ac.EYE
            Rcam = R
            if ac.shake > 0.02 or ac.on_ground:
                amp = 0.0035 * ac.shake + (0.0016 if ac.on_ground and ac.ias > 25 else 0.0)
                Rcam = Rcam @ _jitter(amp, self.ac.time)
            return pos, Rcam
        if self.camera == "chase":
            tgt = ac.pos + R @ np.array((-34.0, 0.0, -8.0))
            if self.chase_pos is None:
                self.chase_pos = tgt
            self.chase_pos = self.chase_pos + (tgt - self.chase_pos) * 0.16
            fwd = ac.pos - self.chase_pos
            return self.chase_pos, _look_dcm(fwd)
        if self.camera == "wing":
            pos = ac.pos + R @ np.array((3.0, -8.0, -1.2))
            return pos, R @ _jitter(0.002, ac.time)
        # tower
        pos = np.array((-300.0, 520.0, -(RWY["elev"] + 48.0)))
        fwd = ac.pos - pos
        return pos, _look_dcm(fwd)

    # ---------------------------------------------------------------- render
    def render(self):
        t0 = time.time()
        r = self.renderer
        ac = self.ac
        pos, R = self.camera_pose()
        r.sky.set(self.sun_elev, self.sun_azim)
        r.set_camera(pos, R)
        r.begin()
        agl = max(0.0, -pos[2] - self.terrain.height(pos[0], pos[1]))
        roll, pitch, yaw = quat_to_euler(ac.quat)
        rings = self.terrain.lod_plan(agl, pitch * RAD - 3.0, self.quality)
        v, t, c, m, b = self.terrain.ring_mesh(pos[0], pos[1], 0, rings=rings)
        r.add_mesh(v, t, c, mat=m, bias=b)
        agd = max(0.0, -pos[2] - 26.0)
        r.fog_range = float(np.clip(15000.0 + 5.0 * agd, 12000.0, 55000.0))
        ld = float(np.hypot(pos[0], pos[1]))          # dist to runway centre
        self.world.draw(r, pos, R, {"clouds": True, "light_d": ld}, self.night())
        if self.camera != "cockpit":
            r.add_mesh(self.world.ac[0], self.world.ac[1], self.world.ac[2],
                       mat=self.world.ac[3], rot=R, pos=ac.pos)
            if ac.gear > 0.5:
                r.add_mesh(self.world.gear[0], self.world.gear[1], self.world.gear[2],
                           mat=self.world.gear[3], rot=R, pos=ac.pos)
            nv = len(self.world.ac[0])
            # nav lights
            from .meshes import Mesh, MAT_UNLIT
            lm = Mesh()
            blink = np.sin(ac.time * 5.0) > 0.75
            lm.box(*(ac.pos + R @ np.array((0.0, -17.6, -2.0))), 0.9, 0.9, 0.5,
                   (255, 40, 40), mat=MAT_UNLIT)
            lm.box(*(ac.pos + R @ np.array((0.0, 17.6, -2.0))), 0.9, 0.9, 0.5,
                   (40, 255, 60), mat=MAT_UNLIT)
            if blink:
                lm.box(*(ac.pos + R @ np.array((-19.0, 0.0, -7.6))), 1.1, 1.1, 0.6,
                       (255, 255, 255), mat=MAT_UNLIT)
            lv, lt, lc, lmm = lm.finalize()
            r.add_mesh(lv, lt, lc, mat=lmm)
        fb = r.draw()
        if self.camera == "cockpit":
            key = (r.w, r.h, r.sky.key)
            if self._cockpit_key != key:
                cv, ct, cc, cm = self.cockpit
                shade = 1.55 - 0.85 * r.sky.night
                cols = np.clip(cc * shade, 0, 255)
                r.bake_cockpit(cv, ct, cols)
                self._cockpit_key = key
            r.apply_cockpit()
        self.frame_ms = (time.time() - t0) * 1000.0
        return fb

    # ----------------------------------------------------------------- state
    def state(self):
        ac = self.ac
        ap = ac.ap
        return {
            "ias": round(ac.ias, 1), "alt": round(-ac.pos[2] * 3.28084),
            "agl": round(max(ac.agl, 0) * 3.28084), "vs": round(ac.vs),
            "hdg": round(ac.hdg), "pitch": round(ac.pitch, 1), "roll": round(ac.roll, 1),
            "thr": round(max(ac.throttle, 0) * 100), "n1": round(ac.n1, 1),
            "flap": ac.FLAP_LBL[ac.flap_detent], "gear": round(ac.gear, 2),
            "spd": round(ac.spoilers, 2), "brk": round(ac.brakes, 1),
            "park": round(ac.park_brake, 1),
            "mach": round(ac.mach, 3), "alpha": round(ac.alpha, 1),
            "fuel": round(ac.fuel), "g": round(ac.gload, 2),
            "ap": 1 if ap.active else 0, "at": 1 if ap.at else 0,
            "rm": ap.roll_mode, "pm": ap.pitch_mode, "app": 1 if ap.app_armed else 0,
            "loc": round(ap.loc_dev, 2), "gs": round(ap.gs_dev, 2),
            "thdg": round(ap.t_hdg), "talt": round(ap.t_alt),
            "tvs": round(ap.t_vs), "tias": round(ap.t_ias),
            "warn": ac.warnings, "sit": ac.situation, "cam": self.camera,
            "fps": round(self.fps, 1), "tris": self.renderer.stats.get("drawn", 0),
            "night": 1 if self.night() else 0, "tod": round(self.time_of_day, 3),
            "res": [self.renderer.w, self.renderer.h],
            "touch": ac.touch_sink,
        }


_JT = np.random.default_rng(5)


def _jitter(amp, t):
    r = np.array((np.sin(t * 13.7), np.sin(t * 11.3 + 2.0), np.sin(t * 17.1 + 4.0))) * amp
    return dcm(euler_to_quat(r[0], r[1], r[2]))
