"""A practical two-axis autopilot + autothrottle with ILS capture.

Modes:  roll  : OFF / HDG / LOC
        pitch : OFF / VS / ALT / IAS / GS
        A/T   : holds IAS (or VS when pitch mode is IAS... kept simple: IAS)
The AP writes control-surface targets; the pilot's inputs are overridden while
a mode is engaged, exactly like a real coupled flight director.
"""
import numpy as np

from .mathx import DEG, RAD, wrap_pi, clamp, dcm, quat_to_euler
from .world import ILS, RWY


class Autopilot:
    def __init__(self, ac):
        self.ac = ac
        self.active = False
        self.at = False
        self.roll_mode = "OFF"
        self.pitch_mode = "OFF"
        self.app_armed = False
        self.t_hdg = 270.0
        self.t_alt = 3000.0
        self.t_vs = 0.0
        self.t_ias = 140.0
        self._de = 0.0
        self._da = 0.0
        self._thr = 0.4
        self._i = 0.4
        self._gsi = 0.0
        self.loc_dev = 0.0
        self.gs_dev = 0.0
        self.annun = ""

    # ------------------------------------------------------------- commands
    def toggle(self):
        self.active = not self.active
        if self.active:
            self.t_hdg = self.ac.hdg
            self.t_alt = -self.ac.pos[2]
            self.t_ias = float(np.clip(self.ac.ias, 110.0, 250.0))
            if self.roll_mode == "OFF":
                self.roll_mode = "HDG"
            if self.pitch_mode == "OFF":
                self.pitch_mode = "ALT"

    def toggle_at(self):
        self.at = not self.at
        if self.at:
            self.t_ias = float(np.clip(self.ac.ias, 110.0, 250.0))
            self._i = float(np.clip(self.ac.throttle, 0.0, 1.0))

    def arm_app(self):
        self.app_armed = not self.app_armed
        if self.app_armed:
            self.active = True
            self.t_ias = min(self.t_ias, 150.0)

    def set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------ step
    def _ils_dev(self):
        ac = self.ac
        x, y, z = ac.pos
        thr = RWY["thr27"]
        # distance along course (west is -Y): remaining distance to threshold
        along = y - thr[1]          # >0 when east of threshold (on approach)
        lat = x - thr[0]            # north positive
        crs = ILS["crs"] * DEG
        # localizer: angular deviation, scaled to ~2.5 deg full scale
        self.loc_dev = clamp(np.degrees(np.arctan2(lat, max(along, 200.0))) / 2.5, -2, 2)
        gs_h = RWY["elev"] + max(along, 0.0) * np.tan(ILS["gs_deg"] * DEG)
        alt = -z
        self.gs_dev = clamp((alt - gs_h) / max(0.02 * max(along, 300.0), 12.0), -2, 2)
        return along

    def step(self, dt):
        ac = self.ac
        if not self.active and not self.at:
            return
        along = self._ils_dev()
        ann = []
        # mode arming logic
        if self.app_armed:
            if self.roll_mode != "LOC" and abs(self.loc_dev) < 1.2 and along < 22000 and along > 0:
                self.roll_mode = "LOC"
                ann.append("LOC")
            if self.pitch_mode != "GS" and self.roll_mode == "LOC" and abs(self.gs_dev) < 1.0 and along < 14000:
                self.pitch_mode = "GS"
                ann.append("GS")
        if along < 200 and self.pitch_mode == "GS":
            self.active = False
            self.app_armed = False
        # ---- roll axis
        if self.active:
            if self.roll_mode == "HDG":
                err = wrap_pi((self.t_hdg - ac.hdg) * DEG)
                bank_t = clamp(err * 1.8, -22 * DEG, 22 * DEG)
            elif self.roll_mode == "LOC":
                bank_t = clamp(-self.loc_dev * 7.0, -14.0, 14.0) * DEG
            else:
                bank_t = 0.0
            bank_err = bank_t - ac.roll * DEG
            da = clamp(bank_err * 1.9 - ac.pqr[0] * 0.9, -0.28, 0.28)
            self._da = da
            # ---- pitch axis
            spd = max(ac.ias, 40) * 0.5144
            if self.pitch_mode == "GS":
                self._gsi = clamp(self._gsi - self.gs_dev * dt * 0.10, -0.05, 0.05)
                gam = -ILS["gs_deg"] - self.gs_dev * 2.2
                if along < 600.0:
                    gam = max(gam, -0.6 - 2.6 * (along / 600.0))
                pitch_t = clamp((ac.alpha + gam) * DEG + self._gsi, -6 * DEG, 20 * DEG)
                pitch_err = pitch_t - ac.pitch * DEG
                self._de = clamp(pitch_err * 1.8 - ac.pqr[1] * 1.4, -0.35, 0.35)
            if self.pitch_mode == "ALT":
                err = self.t_alt - (-ac.pos[2])
                vs_t = clamp(err * 0.055, -9.0, 9.0)
            elif self.pitch_mode == "VS":
                vs_t = self.t_vs / 196.85
            elif self.pitch_mode == "GS":
                vs_t = -spd * np.tan(ILS["gs_deg"] * DEG) - self.gs_dev * 3.0
            elif self.pitch_mode == "IAS":
                vs_t = clamp((self.t_ias - ac.ias) * 0.10, -12.0, 7.0)
            else:
                vs_t = 0.0
            if self.pitch_mode != "GS":
                vs = ac.vs / 196.85
                vs_err = vs_t - vs
                pitch_t = clamp(ac.pitch * DEG + vs_err * 0.035, -7 * DEG, 18 * DEG)
                pitch_err = pitch_t - ac.pitch * DEG
                self._de = clamp(pitch_err * 1.5 - ac.pqr[1] * 1.5, -0.3, 0.3)
        else:
            self._de = 0.0
            self._da = 0.0
        # ---- autothrottle
        if self.at:
            err = self.t_ias - ac.ias
            self._i = clamp(self._i + err * dt * 0.09, 0.0, 1.0)
            self._thr = clamp(self._i + err * 0.03, 0.0, 1.0)
            ac.throttle_cmd = self._thr
        self.annun = " ".join(ann)

    def surface_target(self, tgt):
        out = tgt.copy()
        if self.active:
            out[0] = self._de
            out[1] = self._da
        return out
