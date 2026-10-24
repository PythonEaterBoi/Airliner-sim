"""6-DoF flight model for a narrow-body twinjet (body-axis formulation).

State is integrated in body axes (u,v,w / p,q,r) with a quaternion attitude and
a NED world position.  Aerodynamics are a stability-derivative model with a
post-stall CL blend, ground effect, gear/flap/spoiler increments, plus a
clamped ground contact model with braking and nose-wheel steering.
"""
import numpy as np

from .mathx import (DEG, RAD, quat_integrate, quat_to_euler, dcm, wrap_pi,
                    clamp, euler_to_quat)
from .autopilot import Autopilot

G = 9.80665
RHO0 = 1.225


def atmosphere(h):
    h = max(0.0, h)
    if h < 11000.0:
        T = 288.15 - 0.0065 * h
        p = 101325.0 * (T / 288.15) ** 5.25588
    else:
        T = 216.65
        p = 22632.0 * np.exp(-(h - 11000.0) / 6341.6)
    return T, p, p / (287.058 * T)


class Aircraft:
    # ---- type data (A320/B737 class)
    S = 122.6
    B = 35.8
    C = 4.2
    MASS = 62000.0
    IXX, IYY, IZZ = 1.30e6, 3.90e6, 4.70e6
    THRUST_MAX = 2 * 118000.0
    VMO = 340.0          # kt IAS
    MMO = 0.82
    FLAP_SET = (0.0, 0.12, 0.25, 0.4, 0.6, 0.8, 1.0)   # detents 0,1,2,5?,.. simplified
    FLAP_LBL = ("0", "1", "5", "10", "15", "25", "30")
    VFE = (350, 250, 240, 225, 210, 195, 180)
    GEAR_DOWN_AT = 13.0   # s
    EYE = np.array((14.5, -0.55, -2.05))

    def __init__(self, terrain):
        self.terrain = terrain
        self.stick = np.zeros(2)       # x roll (+right), y pitch (+back)
        self.pedals = 0.0
        self.throttle_cmd = 0.35
        self.throttle = 0.35
        self.trim = 0.0
        self.flap_detent = 3
        self.flaps = 0.4
        self.gear_cmd = 1
        self.gear = 1.0
        self.spoilers = 0.0
        self.spoiler_cmd = 0.0
        self.brakes = 0.0
        self.brake_cmd = 0.0
        self.park_brake = 1.0
        self.surfaces = np.zeros(3)    # elevator, aileron, rudder (deflected)
        self.reset_situation("approach")
        self.ap = Autopilot(self)
        self.hdg = 270.0
        self.pitch = 0.0
        self.roll = 0.0
        self.ias = 0.0
        self.vs = 0.0
        self.mach = 0.0
        self.alpha = 0.0
        self.gload = 1.0
        self.n1 = 22.0
        self.shake = 0.0
        self._cl = 0.3
        self._cl_max = 1.4
        self._qbar = 0.0
        self._rho = RHO0
        self._mach = 0.0
        self.warnings = []
        self.msg = ""
        self.touch_sink = 0.0
        self.on_ground = False
        self.agl = 0.0
        self.ias = 0.0
        self.mach = 0.0
        self.vs = 0.0
        self.alpha = 0.0
        self.gload = 1.0
        self.n1 = 0.0
        self.fuel = 9200.0
        self.time = 0.0
        self.shake = 0.0
        self.events = []

    # ------------------------------------------------------------ situations
    def reset_situation(self, name):
        self.situation = name
        self.stick[:] = 0.0
        self.pedals = 0.0
        self.surfaces[:] = 0.0
        self.park_brake = 1.0
        self.brakes = 0.0
        self.spoilers = 0.0
        self.spoiler_cmd = 0.0
        self.gear_cmd = 1
        self.gear = 1.0
        if name == "takeoff":
            self.pos = np.array((0.0, 1450.0, -28.6))
            self.quat = euler_to_quat(0.0, 0.0, 270.0 * DEG)
            self.vel = np.zeros(3)
            self.pqr = np.zeros(3)
            self.throttle_cmd = 0.0
            self.throttle = 0.0
            self.trim = 0.0
            self.flap_detent = 2
            self.flaps = 0.25
            self.park_brake = 0.0
        elif name == "approach":
            self.pos = np.array((0.0, 11800.0, -640.0))
            self.quat = euler_to_quat(0.0, 5.0 * DEG, 270.0 * DEG)
            spd = 74.0
            self.vel = np.array((spd, 0.0, spd * np.tan(8.5 * DEG)))
            self.trim = 0.10
            self.pqr = np.zeros(3)
            self.throttle_cmd = 0.42
            self.throttle = 0.42
            self.flap_detent = 4
            self.flaps = 0.6
            self.gear_cmd = 1
            self.park_brake = 0.0
        elif name == "cruise":
            self.pos = np.array((-4000.0, 16000.0, -7300.0))
            self.quat = euler_to_quat(0.0, 2.5 * DEG, 255.0 * DEG)
            self.vel = np.array((215.0, 0.0, 215.0 * np.tan(2.5 * DEG)))
            self.trim = 0.03
            self.pqr = np.zeros(3)
            self.throttle_cmd = 0.82
            self.throttle = 0.82
            self.flap_detent = 0
            self.flaps = 0.0
            self.gear_cmd = 0
            self.gear = 0.0
            self.park_brake = 0.0
        else:  # pattern
            self.pos = np.array((-2400.0, -2200.0, -330.0))
            self.quat = euler_to_quat(0.0, 8.0 * DEG, 90.0 * DEG)
            self.vel = np.array((78.0, 0.0, 78.0 * np.tan(9.0 * DEG)))
            self.trim = 0.14
            self.pqr = np.zeros(3)
            self.throttle_cmd = 0.55
            self.throttle = 0.55
            self.flap_detent = 2
            self.flaps = 0.25
            self.gear_cmd = 1
            self.park_brake = 0.0
        self.vel_body0 = self.vel.copy()
        self.ias = float(np.linalg.norm(self.vel)) * 1.94384

    # ------------------------------------------------------------------ step
    def step(self, dt):
        self.time += dt
        self._controls(dt)
        self._aero_and_integrate(dt)
        self._ground(dt)
        self._instruments(dt)
        self.ap.step(dt)

    def _controls(self, dt):
        # engine / high-lift devices lag
        self.throttle += clamp(self.throttle_cmd - self.throttle, -dt / 2.2, dt / 2.2)
        tgt_flaps = self.FLAP_SET[self.flap_detent]
        self.flaps += clamp(tgt_flaps - self.flaps, -dt / 7.0, dt / 7.0)
        gt = 1.0 if self.gear_cmd else 0.0
        self.gear += clamp(gt - self.gear, -dt / self.GEAR_DOWN_AT, dt / self.GEAR_DOWN_AT)
        self.spoilers += clamp(self.spoiler_cmd - self.spoilers, -dt / 1.6, dt / 1.6)
        self.brakes += clamp(self.brake_cmd - self.brakes, -dt / 0.5, dt / 0.5)
        # control surfaces: rate limited, centred by stick
        if abs(self.stick[1]) < 0.06 and not (self.ap.active and self.ap.pitch_mode != "OFF"):
            self.trim = clamp(self.trim - self.pqr[1] * 2.2 * dt, -0.32, 0.32)
        tgt = np.array((self.stick[1] * 0.35 + self.trim,
                        self.stick[0] * 0.42,
                        self.pedals * 0.45))
        if self.ap.active:
            tgt = self.ap.surface_target(tgt)
        rate = np.array((0.6, 1.4, 1.2)) * dt
        self.surfaces += np.clip(tgt - self.surfaces, -rate, rate)
        self.n1 += (self._n1_target() - self.n1) * min(1.0, dt / 2.0)

    def _n1_target(self):
        idle = 22.0
        return idle + (103.0 - idle) * max(0.0, self.throttle)

    # ------------------------------------------------------------------ aero
    def _aero_and_integrate(self, dt):
        R = dcm(self.quat)
        roll, pitch, yaw = quat_to_euler(self.quat)
        u, v, w = self.vel
        V = max(np.sqrt(u * u + v * v + w * w), 1.0)
        alpha = np.arctan2(w, max(u, 1.0))
        beta = np.arcsin(clamp(v / V, -1, 1))
        h = -self.pos[2]
        T, p, rho = atmosphere(h - 0.0)
        rho = max(rho, 0.02)
        qbar = 0.5 * rho * V * V
        a_spd = 340.3 * np.sqrt(T / 288.15)
        mach = V / a_spd

        # lift curve with stall blend
        fl = self.flaps
        a_stall = (13.5 + 4.0 * fl) * DEG
        cl_a = 5.6
        cl0 = 0.16 + 0.72 * fl
        cl_lin = cl0 + cl_a * alpha
        cl_max = 1.35 + 0.85 * fl
        if alpha > a_stall:
            over = (alpha - a_stall) / (25 * DEG)
            cl = cl_max * (1 - 0.55 * min(over, 1.0)) + 0.9 * np.sin(2 * alpha) * min(over, 1) * 0.25
        elif alpha < -a_stall * 0.6:
            cl = cl_lin * 0.5
        else:
            cl = cl_lin
        # ground effect
        agl = self.agl
        ge = 1.0 - 0.45 * np.exp(-max(agl, 0.0) / (0.25 * self.B))
        k_ind = 0.043 * ge
        cd = 0.0215 + k_ind * cl * cl + 0.055 * fl + 0.028 * self.gear + 0.075 * self.spoilers
        cy = -0.55 * beta
        # body-axis aero forces
        ca, sa = np.cos(alpha), np.sin(alpha)
        fx = qbar * self.S * (cl * sa - cd * ca)
        fy = qbar * self.S * cy
        fz = qbar * self.S * (-cl * ca - cd * sa)
        # thrust
        thr = self.THRUST_MAX * (rho / RHO0) * (1.0 - 0.09 * mach) * max(self.throttle, 0.0)
        rev = -0.45 * self.THRUST_MAX * (rho / RHO0) * min(0.0, self.throttle) / -0.45 if self.throttle < 0 else 0.0
        if self.throttle < 0.0:
            thr = self.THRUST_MAX * (rho / RHO0) * 0.4 * self.throttle
        fx += thr
        # moments
        p, q, r = self.pqr
        de, da, dr = self.surfaces
        ph = self.B * p / (2 * V)
        qh = self.C * q / (2 * V)
        rh = self.B * r / (2 * V)
        cl_m = (-0.13 * beta - 0.46 * ph + 0.09 * rh + 0.20 * da + 0.02 * dr)
        cm = (-0.055 - 0.85 * alpha + 0.10 * fl + 1.15 * de - 9.0 * qh - 0.02 * self.spoilers)
        cn = (0.10 * beta - 0.17 * rh + 0.09 * dr - 0.012 * da - 0.02 * ph)
        L = qbar * self.S * self.B * cl_m
        M = qbar * self.S * self.C * cm
        N = qbar * self.S * self.B * cn
        # gravity in body axes
        fg = self.MASS * G * np.array((-np.sin(pitch),
                                       np.sin(roll) * np.cos(pitch),
                                       np.cos(roll) * np.cos(pitch)))
        F = np.array((fx, fy, fz)) + fg
        acc = F / self.MASS - np.cross(self.pqr, self.vel)
        self.vel = self.vel + acc * dt
        I = np.array((self.IXX, self.IYY, self.IZZ))
        mom = np.array((L, M, N)) - np.cross(self.pqr, I * self.pqr)
        self.pqr = self.pqr + (mom / I) * dt
        # damp numerical spin near stall a touch
        self.quat = quat_integrate(self.quat, self.pqr, dt)
        R = dcm(self.quat)
        self.pos = self.pos + (R @ self.vel) * dt
        self.fuel = max(0.0, self.fuel - (0.35 + 2.6 * max(self.throttle, 0)) * dt)
        self._cl = cl
        self._cl_max = cl_max
        self._alpha = alpha
        self._qbar = qbar
        self._rho = rho
        self._mach = mach

    # ---------------------------------------------------------------- ground
    def _ground(self, dt):
        x, y, z = self.pos
        terr = float(self.terrain.height(x, y))
        R = dcm(self.quat)
        roll, pitch, yaw = quat_to_euler(self.quat)
        # gear/keel height below CG
        drop = 3.1 * self.gear + 2.2 * (1.0 - self.gear)
        ground_z = -terr
        agl = -(z) - terr - drop
        self.agl = agl
        vw = R @ self.vel
        self.on_ground = False
        if agl <= 0.0:
            self.on_ground = True
            if vw[2] > 0.5:
                self.touch_sink = vw[2] * 196.85
                self.events.append(("touch", round(self.touch_sink)))
            # put gear on the surface
            self.pos[2] = ground_z - drop
            # remove downward velocity (body frame): project
            down_w = np.array((0.0, 0.0, 1.0))
            down_b = R.T @ down_w
            v_down = float(self.vel @ down_b)
            if v_down > 0.0:
                self.vel = self.vel - down_b * v_down * 1.0
            # vertical bounce for hard landings
            if self.touch_sink > 600:
                self.vel = self.vel + down_b * (-min(self.touch_sink / 196.85, 3.0) * 0.25)
            # friction / braking
            u = self.vel[0]
            N = max(0.0, self.MASS * G - self._qbar * self.S * self._cl)
            mu = 0.02 + 0.42 * self.brakes + 0.5 * self.park_brake
            if self.spoilers > 0.5:
                mu += 0.0
            f_fric = -np.sign(u) * min(abs(u) * 40.0, 1.0) * mu * N
            self.vel[0] += (f_fric / self.MASS) * dt
            # tyres kill side slip and yaw
            self.vel[1] *= np.exp(-4.0 * dt)
            self.pqr[2] *= np.exp(-2.5 * dt)
            self.pqr[0] *= np.exp(-6.0 * dt)
            # steering
            steer = self.pedals * 0.55 + self.stick[0] * 0.10
            spd = abs(u)
            if spd > 1.0:
                target_r = steer * np.clip(1.6 - spd / 60.0, 0.12, 1.6) * 0.55
                self.pqr[2] += (target_r - self.pqr[2]) * min(1.0, 6 * dt)
            # pitch: settle on gear, allow rotation above Vr
            vr = 62.0 + 8.0 * self.flaps
            if spd < vr and pitch < 0.0:
                self.pqr[1] += (0.0 - pitch) * min(1.0, 4 * dt) * 0.4
            self.pqr[1] *= np.exp(-1.5 * dt)
            if pitch > 13 * DEG:
                self.pqr[1] -= (pitch - 13 * DEG) * 6 * dt
            if pitch < -2 * DEG:
                self.pqr[1] += (-2 * DEG - pitch) * 6 * dt
            self.throttle = max(self.throttle, 0.0)
        else:
            self.touch_sink = 0.0

    # ----------------------------------------------------------- instruments
    def _instruments(self, dt):
        R = dcm(self.quat)
        vw = R @ self.vel
        V = max(np.linalg.norm(self.vel), 0.1)
        T, p, rho = atmosphere(-self.pos[2])
        qc = 0.5 * self._rho * V * V
        ias = np.sqrt(2 * qc / RHO0)
        self.ias = ias * 1.94384
        self.mach = self._mach
        self.vs = -vw[2] * 196.85
        self.alpha = self._alpha * RAD
        self.gload = float((R.T @ np.array((0, 0, -G))) @ np.array((0, 0, -1))) / G + \
            float((R @ np.array((0, 0, 0))) @ np.array((0, 0, 0)))
        az = (R @ (np.cross(self.pqr, self.vel) * -1 + np.array((0, 0, 0))))
        self.gload = clamp(self._qbar * self.S * self._cl / (self.MASS * G), -2, 4)
        self.hdg = (quat_to_euler(self.quat)[2] * RAD) % 360.0
        self.pitch = quat_to_euler(self.quat)[1] * RAD
        self.roll = quat_to_euler(self.quat)[0] * RAD
        # buffet near stall
        margin = (self._cl_max - self._cl) / max(self._cl_max, 0.1)
        self.shake = clamp(1.0 - margin / 0.12, 0, 1) if self._cl > 0.3 else 0.0
        self._warnings()

    def _warnings(self):
        w = []
        if self.on_ground:
            pass
        else:
            if self.shake > 0.35 or (self.alpha > 13.0 and self.ias > 40):
                w.append("STALL")
            if self.ias > self.VMO or self.mach > self.MMO:
                w.append("OVERSPEED")
            if self.vs < -1800 and self.agl < 1200:
                w.append("SINK RATE")
            if self.vs < -2400 and self.agl < 900:
                w.append("PULL UP")
            if self.agl < 600 and self.agl > 60 and self.gear < 0.9:
                w.append("TOO LOW GEAR")
            if self.agl < 400 and self.ias > self.VFE[self.flap_detent]:
                w.append("FLAPS")
        if self.ias > self.VFE[self.flap_detent] + 5:
            w.append("FLAP LIM")
        if self.gear > 0.1 and self.ias > 255:
            w.append("GEAR SPEED")
        self.warnings = w
