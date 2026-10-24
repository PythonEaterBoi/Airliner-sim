"""The world around the airfield: runway complex, buildings, city, lights,
clouds and AI traffic.  Static geometry is built once; per-frame the pieces
are bounding-sphere culled and handed to the renderer.

Runway 09/27 lies along the east (Y) axis; RWY 27 threshold is at +Y, so an
approach to 27 comes in over the ocean from the east heading west (270 deg).
"""
import numpy as np

from .mathx import Rng, DEG, wrap_pi
from .meshes import (Mesh, airliner_mesh, gear_mesh, cloud_mesh, tube,
                     MAT_LAND, MAT_UNLIT)
from .terrain import CITY, AIRPORT_ELEV

RWY = {
    "len": 3200.0,
    "wid": 45.0,
    "elev": AIRPORT_ELEV,
    "thr27": np.array((0.0, 1600.0)),
    "thr09": np.array((0.0, -1600.0)),
    "crs27": 270.0,
    "crs09": 90.0,
}
ILS = {
    "crs": 270.0,
    "loc": np.array((0.0, 2950.0)),
    "gs": np.array((-125.0, 1250.0)),
    "gs_deg": 3.0,
    "freq": "110.3",
}


class Piece:
    __slots__ = ("data", "center", "radius", "min_range", "name")

    def __init__(self, name, data, min_range=0.0):
        v = data[0]
        self.data = data
        self.center = (v.min(axis=0) + v.max(axis=0)) * 0.5
        self.radius = float(np.max(np.linalg.norm(v - self.center, axis=1))) + 1.0
        self.min_range = min_range
        self.name = name


def _runway_complex():
    m = Mesh()
    e = RWY["elev"]
    asphalt = (62, 63, 66)
    conc = (122, 124, 126)
    white = (226, 226, 220)
    yel = (196, 168, 40)
    hw = RWY["wid"] * 0.5
    L = RWY["len"] * 0.5
    # runway slab
    m.ground_quad(-hw, hw, -L, L, e, asphalt, bias=0.25)
    # edge stripes
    m.ground_quad(-hw, -hw + 0.9, -L, L, e, white, bias=0.32)
    m.ground_quad(hw - 0.9, hw, -L, L, e, white, bias=0.32)
    # centreline dashes
    y = -L + 60.0
    while y < L - 60.0:
        m.ground_quad(-0.45, 0.45, y, y + 30.0, e, white, bias=0.32)
        y += 50.0
    # threshold bars + numbers + aiming points, both ends
    for sgn, num, uy in ((1, "27", (0.0, -1.0)), (-1, "09", (0.0, 1.0))):
        y_thr = sgn * L
        for i in range(8):
            x = -hw + 3.0 + i * (RWY["wid"] - 6.0) / 7.0 - 1.0
            y0 = y_thr - sgn * 30.0
            m.ground_quad(x, x + 1.8, min(y0, y_thr - sgn * 0.5), max(y0, y_thr - sgn * 0.5),
                          e, white, bias=0.32)
        # touchdown zone
        for d in (150.0, 450.0, 750.0):
            for sx in (-1, 1):
                x0 = sx * 12.0 - (4.5 if sx < 0 else -4.5) - 4.5
                m.ground_quad(sx * 9.0 - 4.5, sx * 9.0 + 4.5,
                              y_thr - sgn * (d + 18.0), y_thr - sgn * d, e, white, bias=0.32)
        m.text_flat(num, 0.0, y_thr - sgn * 120.0, e, 12.0, ux=(1.0, 0.0), uy=uy, bias=0.32)
    # aiming point bars
    for sgn in (1, -1):
        for sx in (-1, 1):
            m.ground_quad(sx * 11.0 - 3.0, sx * 11.0 + 3.0, sgn * (L - 1050.0),
                          sgn * (L - 1000.0), e, white, bias=0.32)
    # parallel taxiway + connectors + apron
    m.ground_quad(-121.0, -98.0, -1450.0, 1450.0, e, conc, bias=0.2)
    m.ground_quad(-116.0, -104.0, -1450.0, 1450.0, e, (150, 150, 140), bias=0.26)
    for yc in (-1200.0, -400.0, 400.0, 1200.0):
        m.ground_quad(-98.0, -hw, yc - 11.0, yc + 11.0, e, conc, bias=0.2)
    m.ground_quad(-330.0, -121.0, -700.0, 500.0, e, conc, bias=0.18)
    m.ground_quad(-330.0, -121.0, -700.0, 500.0, e, (112, 114, 116), bias=0.22)
    # apron lead-in lines
    for yc in (-500.0, -100.0, 300.0):
        m.ground_quad(-300.0, -121.0, yc - 0.5, yc + 0.5, e, yel, bias=0.26)
    return m.finalize()


def _lights(mode=0):
    """Runway / threshold / approach lights.

    mode 0: flat 1 m ground quads (close range, < 2.5 km)
    mode 1: upright bars 8 x 5 m   (2.5 - 6.5 km)
    mode 2: upright bars 26 x 14 m (> 6.5 km, stays ~1 px at 10 km)
    Ground quads foreshorten to nothing at glide-slope angles, so the far
    LODs stand the lights upright instead.
    """
    m = Mesh()
    e = RWY["elev"]
    L = RWY["len"] * 0.5
    hw = RWY["wid"] * 0.5
    w = (255, 250, 235)
    g = (60, 255, 90)
    r = (255, 45, 40)
    bw, bh = ((1.5, 1.0) if mode == 0 else
              (12.0, 6.0) if mode == 1 else (30.0, 16.0))

    def put(cx, cy, hx, hy, col, bias=0.3):
        m.ground_quad(cx - hx, cx + hx, cy - hy, cy + hy, e, col,
                      mat=MAT_UNLIT, bias=bias)
        m.quad((cx - bw, cy, -e), (cx + bw, cy, -e),
               (cx + bw, cy, -e - bh * 2), (cx - bw, cy, -e - bh * 2),
               col, mat=MAT_UNLIT, double=True)

    y = -L
    while y <= L + 1e-6:
        put(-hw - 1.25, y, 0.35, 0.5, w)
        put(hw + 1.25, y, 0.35, 0.5, w)
        y += 60.0
    for sgn, end_col, thr_col in ((1, r, g), (-1, g, r)):
        y_thr = sgn * L
        for i in range(10):
            x = -hw + 2.0 + i * (RWY["wid"] - 4.0) / 9.0
            put(x, y_thr - sgn * 0.7, 0.5, 0.5, thr_col, 0.34)
            put(x, y_thr + sgn * 0.7, 0.5, 0.5, end_col, 0.34)
        # ALSF style approach lights
        d = 30.0
        step = 30.0 if mode < 2 else 60.0
        while d < 900.0:
            yc = y_thr + sgn * d
            wide = 5 if d > 300 else 3
            for i in range(wide):
                x = (i - (wide - 1) / 2.0) * (10.0 if d > 300 else 6.0)
                put(x, yc, 0.5, 0.5, w)
            if 240 < d < 900:
                for i in range(2):
                    x = (i - 0.5) * 20.0
                    put(x, yc + sgn * 8.4, 0.4, 0.4, r)
            d += step
    if mode == 0:
        # taxiway edge (blue) hints, close range only
        for yc in np.arange(-1440.0, 1441.0, 120.0):
            m.ground_quad(-122.0, -121.2, yc - 0.4, yc + 0.4, e, (60, 120, 255),
                          mat=MAT_UNLIT, bias=0.28)
            m.ground_quad(-97.8, -97.0, yc - 0.4, yc + 0.4, e, (60, 120, 255),
                          mat=MAT_UNLIT, bias=0.28)
    return m.finalize()


def _airport_buildings():
    m = Mesh()
    e = RWY["elev"]
    m.box(-260, -100, e, 210, 95, 16, (168, 170, 174), top=(150, 152, 156))
    m.box(-260, -100, e + 16, 150, 60, 5, (150, 152, 156))
    m.box(-330, -100, e, 40, 300, 12, (160, 162, 166))       # pier
    m.box(-300, 520, e, 14, 14, 42, (196, 196, 192))          # tower shaft
    m.box(-300, 520, e + 42, 20, 20, 7, (40, 60, 70), top=(120, 130, 135))
    m.box(-250, -950, e, 95, 75, 15, (150, 148, 142), top=(120, 118, 114))
    m.box(-250, -1250, e, 95, 75, 15, (150, 148, 142), top=(120, 118, 114))
    for i, (dx, dy) in enumerate(((-360, 850), (-360, 900), (-390, 850), (-390, 900))):
        tube(m, (dx, dy, -e), (dx, dy, -(e + 14)), 9.0, 9.0, sides=8, color=(206, 206, 200))
    m.box(-150, 2350, e, 8, 30, 3, (180, 180, 176))           # localizer shack
    for i in range(12):
        m.box(-2.0 + (i % 2) * 4.0, 2800 + i * 12, e, 3.4, 1.0, 4.2, (170, 172, 176))
    m.box(-125, 1250, e, 6, 6, 8, (180, 180, 176))            # glideslope
    return m.finalize()


def _city():
    rng = Rng(seed=4242)
    m = Mesh()
    gx = np.arange(-6, 7)
    gy = np.arange(-6, 7)
    palette = ((146, 142, 136), (128, 130, 138), (162, 156, 148), (110, 114, 122),
               (176, 170, 160), (96, 100, 108), (138, 128, 118))
    h0 = 18.0
    # urban ground plate + street grid so downtown reads as a city, not boxes
    m.ground_quad(CITY[0] - 2950, CITY[0] + 2950, CITY[1] - 2950, CITY[1] + 2950,
                  h0 + 0.35, (84, 88, 82))
    for i in gx:
        xo = CITY[0] + i * 420.0 + 210.0
        yo = CITY[1] + i * 420.0 + 210.0
        m.ground_quad(xo - 15, xo + 15, CITY[1] - 2950, CITY[1] + 2950, h0 + 0.7,
                      (58, 58, 61))
        m.ground_quad(CITY[0] - 2950, CITY[0] + 2950, yo - 15, yo + 15, h0 + 0.7,
                      (58, 58, 61))
    for i in gx:
        for j in gy:
            if rng.rng.random() < 0.22:
                continue
            cx = CITY[0] + i * 420.0 + rng.uniform(1, -60, 60)
            cy = CITY[1] + j * 420.0 + rng.uniform(1, -60, 60)
            d = np.hypot(i, j)
            tall = max(0.15, 1.0 - d / 7.5)
            hh = 10.0 + rng.uniform(1, 0, 1) * 70.0 * tall + (55.0 if rng.rng.random() < 0.06 else 0.0)
            sx = rng.uniform(1, 26, 62)
            sy = rng.uniform(1, 26, 62)
            col = palette[int(rng.rng.integers(0, len(palette)))]
            m.box(cx, cy, h0, sx, sy, hh, col,
                  top=tuple(min(255, int(c * 1.2)) for c in col))
    # a couple of landmark towers
    m.box(CITY[0] + 200, CITY[1] + 300, h0, 46, 46, 148, (120, 132, 148))
    m.box(CITY[0] - 600, CITY[1] - 400, h0, 40, 40, 122, (150, 140, 128))
    # ring road
    m.ground_quad(CITY[0] - 3200, CITY[0] + 3200, CITY[1] - 20, CITY[1] + 20, h0 + 0.7, (70, 70, 72))
    m.ground_quad(CITY[0] - 20, CITY[0] + 20, CITY[1] - 3200, CITY[1] + 3200, h0 + 0.7, (70, 70, 72))
    return m.finalize()


class Traffic:
    def __init__(self, idx):
        self.idx = idx
        self.pos = np.zeros(3)
        self.hdg = 0.0
        self.alt = 0.0
        self.spd = 70.0
        self.phase = 0.0
        self.mode = "pattern"

    def update(self, t, dt):
        if self.mode == "pattern":
            # rectangular pattern around RWY 27, 1000 ft
            self.alt = 320.0
            self.spd = 68.0
            p = (t * 0.055 + self.phase) % 4.0
            if p < 1.0:      # downwind (east of field, heading 090)
                x = -1500.0 - 900.0 * min(1.0, p)
                y = -2200.0 + 4200.0 * p
                self.hdg = 90.0
            elif p < 2.0:    # base
                q = p - 1.0
                x = -2400.0 + 2400.0 * q
                y = 2000.0
                self.hdg = 90.0 - 90.0 * min(1.0, q * 3)
            elif p < 3.0:    # final
                q = p - 2.0
                x = 0.0
                y = 2000.0 - 3600.0 * q
                self.hdg = 270.0
                self.alt = 320.0 * (1 - q) + 40.0
            else:            # climb out / crosswind
                q = p - 3.0
                x = 0.0
                y = -1600.0 - 3000.0 * q
                self.hdg = 270.0 + 90.0 * min(1.0, q * 2.5)
                self.alt = 40.0 + 500.0 * q
            self.pos = np.array((x, y, -self.alt))
        elif self.mode == "depart":
            q = (t * 0.02 + self.phase) % 1.0
            d = q * 42000.0
            self.pos = np.array((0.0, 1600.0 - d, -(60.0 + q * 6800.0)))
            self.hdg = 270.0
            self.spd = 120.0 + q * 120.0
        else:  # cruise across the sky
            q = (t * 0.012 + self.phase) % 1.0
            x = -30000.0 + 60000.0 * q
            y = 14000.0 + 4000.0 * np.sin(q * 6.28)
            self.pos = np.array((x, y, -8200.0))
            self.hdg = 90.0
            self.spd = 230.0


class World:
    def __init__(self, terrain, seed=99):
        self.terrain = terrain
        self.pieces = []
        self.pieces.append(Piece("rwy", _runway_complex()))
        self.pieces.append(Piece("lights0", _lights(0)))
        self.pieces.append(Piece("lights1", _lights(1)))
        self.pieces.append(Piece("lights2", _lights(2)))
        self.pieces.append(Piece("aport", _airport_buildings()))
        self.pieces.append(Piece("city", _city()))
        rng = Rng(seed=seed)
        self.clouds = cloud_mesh(rng)
        self.pieces.append(Piece("clouds", self.clouds))
        self.ac = airliner_mesh(detail=1)
        self.ac_lo = airliner_mesh(detail=0)
        self.gear = gear_mesh()
        self.traffic = []
        for i, (mode, ph) in enumerate((("pattern", 0.0), ("depart", 0.35), ("cruise", 0.6))):
            tr = Traffic(i)
            tr.mode = mode
            tr.phase = ph
            self.traffic.append(tr)
        self.t = 0.0
        # parked aircraft transforms
        self.parked = [((-250.0, -500.0, -(AIRPORT_ELEV + 2.6)), 92.0),
                       ((-250.0, -100.0, -(AIRPORT_ELEV + 2.6)), 88.0),
                       ((-250.0, 300.0, -(AIRPORT_ELEV + 2.6)), 91.0)]

    def update(self, dt):
        self.t += dt
        for tr in self.traffic:
            tr.update(self.t, dt)

    # ------------------------------------------------------------------ draw
    def _visible(self, piece, cam, fwd, right, upv, margin=0.35):
        d = piece.center - cam
        z = float(d @ fwd)
        if z < -piece.radius:
            return False
        if z > 70000.0:
            return False
        x = float(d @ right)
        y = float(d @ upv)
        rr = piece.radius + margin * max(z, 1.0)
        return abs(x) < rr + 1.4 * max(z, 1.0) and abs(y) < rr + 1.1 * max(z, 1.0)

    def draw(self, r, cam, R, state, night):
        fwd = R[:, 0]
        right = R[:, 1]
        upv = -R[:, 2]
        for pc in self.pieces:
            if pc.name == "clouds" and not state.get("clouds", True):
                continue
            if pc.name.startswith("lights"):
                tier = 0 if state.get("light_d", 0) < 2500 else \
                       1 if state.get("light_d", 0) < 6500 else 2
                if pc.name != "lights%d" % tier:
                    continue
            if not self._visible(pc, cam, fwd, right, upv):
                continue
            v, t, c, mm = pc.data
            if pc.name == "clouds" and np.asarray(c).ndim == 2:
                c = (np.asarray(c, np.float32) *
                     (0.16 + 0.84 * (1.0 - night))).astype(np.uint8)
            r.add_mesh(v, t, c, mat=mm)
        # parked airliners
        for (pos, hdg) in self.parked:
            d = np.asarray(pos, float) - cam
            if d @ fwd < -60:
                continue
            rot = _yaw_dcm(hdg)
            r.add_mesh(self.ac_lo[0], self.ac_lo[1], self.ac_lo[2], mat=self.ac_lo[3],
                       rot=rot, pos=pos)
        # traffic
        for tr in self.traffic:
            d = tr.pos - cam
            z = d @ fwd
            if z < -80 or z > 40000:
                continue
            rot = _yaw_dcm(tr.hdg)
            mesh = self.ac_lo if z > 6000 else self.ac
            r.add_mesh(mesh[0], mesh[1], mesh[2], mat=mesh[3], rot=rot, pos=tr.pos)
            if night or True:
                bm = Mesh()
                col = (255, 60, 50) if np.sin(self.t * 4.0 + tr.idx) > 0.6 else (18, 18, 18)
                bm.box(tr.pos[0], tr.pos[1], -tr.pos[2] - 8.0, 1.2, 1.2, 0.6, col, mat=MAT_UNLIT)
                bv, bt, bc, bmm = bm.finalize()
                r.add_mesh(bv, bt, bc, mat=bmm)
        # PAPI lights, coloured for the current camera glide path
        self._draw_papi(r, cam)

    def _draw_papi(self, r, cam):
        for (thr, sgn, ux) in ((RWY["thr27"], 1, -1), (RWY["thr09"], -1, 1)):
            base = np.array((thr[0], thr[1] - sgn * 300.0, -RWY["elev"]))
            d = base - cam
            if np.linalg.norm(d) > 12000.0:
                continue
            # glide angle from camera to the threshold aiming point
            horiz = np.hypot(thr[0] - cam[0], thr[1] - cam[1])
            vert = (-RWY["elev"]) - cam[2]
            ga = np.degrees(np.arctan2(vert, horiz))
            m = Mesh()
            for i in range(4):
                ang = 3.0 + (1.5 - i) * 0.55
                col = (255, 40, 40) if ga > ang else (255, 250, 245)
                x = ux * (30.0 + i * 5.0)
                m.ground_quad(base[0] + x - 2.0, base[0] + x + 2.0,
                              base[1] - 2.0, base[1] + 2.0, RWY["elev"], col,
                              mat=MAT_UNLIT, bias=0.4)
            v, t, c, mm = m.finalize()
            r.add_mesh(v, t, c, mat=mm)


def _yaw_dcm(hdg_deg):
    """Body->world DCM for a level aircraft at heading (deg)."""
    h = hdg_deg * DEG
    return np.array((
        (np.cos(h), -np.sin(h), 0.0),
        (np.sin(h), np.cos(h), 0.0),
        (0.0, 0.0, 1.0),
    ), dtype=np.float64)
