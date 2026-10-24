"""Mesh builders and the 3-D models used by the world.

Winding convention: a triangle is front-facing when its vertices run
counter-clockwise as seen from outside (same rule the terrain uses, which the
renderer's cull test expects).  Builders take an outward normal hint or a
`double` flag for thin sheets so this is handled automatically.

World/model frame: NED-ish, X forward/north, Y right/east, Z down.
"""
import numpy as np

MAT_LAND = 0
MAT_WATER = 1
MAT_UNLIT = 2

UP = (0.0, 0.0, -1.0)

FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def _orient(pts, n):
    if n is None:
        return pts
    p0, p1, p2, p3 = pts
    cr = np.cross(np.subtract(p1, p0), np.subtract(p2, p0))
    if np.dot(cr, n) < 0:
        return (p0, p3, p2, p1)
    return pts


class Mesh:
    def __init__(self):
        self.v = []
        self.t = []
        self.c = []
        self.m = []

    def tri(self, p0, p1, p2, color, mat=MAT_LAND, n=None, double=False):
        pts = (p0, p1, p2)
        if n is not None:
            cr = np.cross(np.subtract(p1, p0), np.subtract(p2, p0))
            if np.dot(cr, n) < 0:
                pts = (p0, p2, p1)
        self._emit(pts, color, mat)
        if double:
            self._emit((pts[0], pts[2], pts[1]), color, mat)

    def _emit(self, pts, color, mat):
        i = len(self.v) // 3
        for p in pts:
            self.v.extend(p)
        if len(pts) == 3:
            self.t.append((i, i + 1, i + 2))
            self.c.append(color)
            self.m.append(mat)
        else:
            self.t.append((i, i + 1, i + 2))
            self.t.append((i, i + 2, i + 3))
            self.c.extend((color, color))
            self.m.extend((mat, mat))

    def quad(self, p0, p1, p2, p3, color, mat=MAT_LAND, n=None, double=False):
        pts = _orient((p0, p1, p2, p3), n)
        self._emit(pts, color, mat)
        if double:
            self._emit((pts[0], pts[3], pts[2], pts[1]), color, mat)

    def ground_quad(self, x0, x1, y0, y1, up, color, mat=MAT_LAND, bias=0.0):
        z = -(up + bias)
        self.quad((x0, y0, z), (x0, y1, z), (x1, y1, z), (x1, y0, z), color, mat, n=UP)

    def box(self, cx, cy, up, sx, sy, sz, color, top=None, mat=MAT_LAND):
        z0 = -up
        z1 = -(up + sz)
        x0, x1 = cx - sx * 0.5, cx + sx * 0.5
        y0, y1 = cy - sy * 0.5, cy + sy * 0.5
        tc = top if top is not None else tuple(min(255, int(c * 1.14)) for c in color)
        sc = tuple(int(c * 0.88) for c in color)
        sc2 = tuple(int(c * 0.70) for c in color)
        self.quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), tc, mat, n=UP)
        self.quad((x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1), sc2, mat, n=(-1, 0, 0))
        self.quad((x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0), sc, mat, n=(1, 0, 0))
        self.quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), sc, mat, n=(0, -1, 0))
        self.quad((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), sc2, mat, n=(0, 1, 0))

    def text_flat(self, s, cx, cy, up, h, color=(232, 232, 226), ux=(1.0, 0.0),
                  uy=(0.0, 1.0), mat=MAT_LAND, bias=0.0):
        """Glyph string runs along ux, glyph "up" along uy (both horizontal)."""
        cw = h / 7.0
        total = len(s) * cw * 6.0
        px = -total * 0.5
        for ch in s:
            g = FONT.get(ch.upper(), FONT[" "])
            for r in range(7):
                for c in range(5):
                    if g[r][c] == "1":
                        lx0 = px + c * cw
                        lx1 = lx0 + cw * 0.9
                        ly0 = r * (h / 7.0)
                        ly1 = ly0 + (h / 7.0) * 0.9
                        pts = []
                        for (lx, ly) in ((lx0, ly0), (lx0, ly1), (lx1, ly1), (lx1, ly0)):
                            pts.append((cx + lx * ux[0] + ly * uy[0],
                                        cy + lx * ux[1] + ly * uy[1],
                                        -(up + bias)))
                        self.quad(pts[0], pts[1], pts[2], pts[3], color, mat, n=UP)
            px += cw * 6.0

    def orient_away(self, center, start=0):
        """Flip triangles (from index `start`) so normals point away from centre."""
        v = np.asarray(self.v, np.float32).reshape(-1, 3)
        c = np.asarray(center, np.float32)
        for k in range(start, len(self.t)):
            a, b, cc = self.t[k]
            cr = np.cross(v[b] - v[a], v[cc] - v[a])
            cen = (v[a] + v[b] + v[cc]) / 3.0
            if np.dot(cr, cen - c) < 0:
                self.t[k] = (a, cc, b)

    def finalize(self):
        return (np.asarray(self.v, np.float32).reshape(-1, 3),
                np.asarray(self.t, np.int32).reshape(-1, 3),
                np.asarray(self.c, np.float32).reshape(-1, 3),
                np.asarray(self.m, np.uint8))


def tube(m, p0, p1, r0, r1, sides=8, color=(200, 200, 205), caps=True):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    d = d / max(L, 1e-9)
    ref = np.array((0.0, 0.0, 1.0))
    if abs(np.dot(ref, d)) > 0.9:
        ref = np.array((1.0, 0.0, 0.0))
    a = np.cross(ref, d)
    a /= np.linalg.norm(a)
    b = np.cross(d, a)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    base = np.cos(ang)[:, None] * a[None, :] + np.sin(ang)[:, None] * b[None, :]
    ring0 = p0[None, :] + base * r0
    ring1 = p1[None, :] + base * r1
    start = len(m.t)
    for i in range(sides):
        j = (i + 1) % sides
        m.tri(tuple(ring0[i]), tuple(ring0[j]), tuple(ring1[j]), color)
        m.tri(tuple(ring0[i]), tuple(ring1[j]), tuple(ring1[i]), color)
    if caps:
        m.tri(tuple(p0), tuple(ring0[1]), tuple(ring0[0]), color)
        m.tri(tuple(p0), tuple(ring0[2]), tuple(ring0[1]), color)
        m.tri(tuple(p1), tuple(ring1[0]), tuple(ring1[1]), color)
        m.tri(tuple(p1), tuple(ring1[1]), tuple(ring1[2]), color)
    m.orient_away((p0 + p1) * 0.5, start)


def airliner_mesh(detail=1):
    """Narrow-body twinjet.  Model space: x fwd, y right, z down (metres)."""
    m = Mesh()
    fuse = (226, 228, 232)
    belly = (176, 180, 188)
    acc = (22, 88, 160)
    sides = 8 if detail else 6
    secs = 12 if detail else 8
    xs = np.linspace(-19.0, 19.5, secs)
    prof = []
    for x in xs:
        if x < -16.0:
            r = 1.55 * (1.0 - (abs(x + 16.0) / 3.6) ** 1.6) + 0.18
        elif x < 12.0:
            r = 1.98
        else:
            t = (x - 12.0) / 7.5
            r = 1.98 * np.sqrt(max(0.0, 1.0 - t * t * 0.94)) + 0.12
        zc = -0.10 - max(0.0, x - 12.0) * 0.10 - max(0.0, -16.0 - x) * 0.16
        prof.append((x, max(r, 0.12), zc))
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    start = len(m.t)
    rings = []
    for (x, r, zc) in prof:
        ring = np.stack((np.full(sides, x), np.cos(ang) * r, zc + np.sin(ang) * r * 0.98), 1)
        rings.append(ring)
    for s in range(secs - 1):
        for i in range(sides):
            j = (i + 1) % sides
            za = rings[s][i][2]
            col = fuse if za < 0.55 else belly
            m.tri(tuple(rings[s][i]), tuple(rings[s][j]), tuple(rings[s + 1][j]), col)
            m.tri(tuple(rings[s][i]), tuple(rings[s + 1][j]), tuple(rings[s + 1][i]), col)
    m.orient_away((0, 0, 0), start)
    # cheatline
    for s in range(2, secs - 3):
        x0, x1 = prof[s][0], prof[s + 1][0]
        r = prof[s][1] * 0.99
        m.quad((x0, -r, 0.30), (x1, -r, 0.30), (x1, -r, 0.78), (x0, -r, 0.78), acc,
               n=(0, -1, 0))
        m.quad((x0, r, 0.30), (x0, r, 0.78), (x1, r, 0.78), (x1, r, 0.30), acc, n=(0, 1, 0))
    # wings
    for sgn in (-1, 1):
        nrm = (0, 0, -1.0)
        m.quad((2.6, 1.6 * sgn, 0.10), (10.6, 17.4 * sgn, -2.1), (13.4, 17.4 * sgn, -1.95),
               (8.6, 1.6 * sgn, 0.16), (214, 216, 220), n=nrm, double=True)
        m.quad((8.6, 1.6 * sgn, 0.16), (13.4, 17.4 * sgn, -1.95), (13.4, 17.4 * sgn, -1.6),
               (8.6, 1.6 * sgn, 0.6), (196, 198, 204), n=(0, -sgn, 0), double=True)
        m.quad((10.6, 17.4 * sgn, -2.1), (11.5, 18.3 * sgn, -4.7), (13.9, 18.3 * sgn, -4.6),
               (13.4, 17.4 * sgn, -1.95), acc, n=(0, sgn, 0), double=True)
    # tail
    m.quad((-13.0, 0, -0.6), (-17.6, 0, -7.4), (-19.6, 0, -7.2), (-18.4, 0, -0.4), acc,
           n=(0, 1, 0), double=True)
    for sgn in (-1, 1):
        m.quad((-14.6, 0.4 * sgn, -0.2), (-18.6, 6.4 * sgn, -0.5), (-20.6, 6.2 * sgn, -0.4),
               (-19.2, 0.4 * sgn, -0.1), (210, 212, 216), n=UP, double=True)
    # engines
    for sgn in (-1, 1):
        tube(m, (4.8, 5.2 * sgn, 1.6), (0.4, 5.2 * sgn, 1.6), 1.18, 1.34,
             sides=8 if detail else 6, color=(188, 190, 196))
        m.quad((0.4, 5.2 * sgn - 1.1, 0.5), (0.4, 5.2 * sgn + 1.1, 0.5),
               (0.4, 5.2 * sgn + 1.1, 2.7), (0.4, 5.2 * sgn - 1.1, 2.7), (36, 36, 40),
               n=(-1, 0, 0))
    return m.finalize()


def gear_mesh():
    m = Mesh()
    dark = (58, 60, 64)
    tyre = (28, 28, 30)
    m.box(13.0, 0.0, -6.5, 0.5, 0.5, 2.7, dark)
    m.box(13.0, 0.0, -6.9, 0.6, 1.2, 0.8, tyre)
    for sgn in (-1, 1):
        m.box(4.6, 2.5 * sgn, -6.4, 0.6, 0.6, 3.0, dark)
        m.box(4.6, 2.5 * sgn, -7.0, 0.8, 2.0, 1.0, tyre)
    return m.finalize()


def cockpit_mesh():
    """View-space cockpit shell (x right, y down, z forward), metres.
    Sized so the glareshield sits ~11 deg below the eyepoint: most of the
    frame stays outside world."""
    m = Mesh()
    dark = (40, 42, 46)
    mid = (60, 62, 66)
    panel = (30, 31, 34)
    frame = (52, 54, 58)
    gs = (34, 35, 38)
    # glareshield + main panel: top edge ~18 deg below the sight line so a
    # 3 deg glide path stays fully visible over the panel
    m.quad((-1.05, 0.50, 1.16), (1.05, 0.50, 1.16), (1.05, 0.95, 1.02),
           (-1.05, 0.95, 1.02), panel, n=(0, -0.3, -1), double=True)
    m.quad((-1.12, 0.40, 1.22), (1.12, 0.40, 1.22), (1.05, 0.50, 1.16), (-1.05, 0.50, 1.16),
           gs, n=(0, -1, -0.3), double=True)
    # side walls / consoles
    m.quad((-1.55, 0.1, 0.9), (-1.15, 0.26, 1.1), (-1.15, 0.95, 1.0), (-1.55, 0.9, 0.8), mid,
           n=(1, 0, 0), double=True)
    m.quad((1.55, 0.1, 0.9), (1.55, 0.9, 0.8), (1.15, 0.95, 1.0), (1.15, 0.26, 1.1), mid,
           n=(-1, 0, 0), double=True)
    # centre pedestal
    m.quad((-0.28, 0.74, 1.02), (0.28, 0.74, 1.02), (0.3, 1.24, 0.82), (-0.3, 1.24, 0.82), mid,
           n=UP, double=True)
    # throttle levers
    for xo in (-0.12, 0.0, 0.12):
        m.quad((xo - 0.022, 0.68, 0.92), (xo + 0.022, 0.68, 0.92), (xo + 0.022, 0.60, 0.74),
               (xo - 0.022, 0.60, 0.74), (18, 18, 20), n=(0, -1, 0), double=True)
    # yokes
    for xo in (-0.62, 0.62):
        m.quad((xo - 0.17, 0.80, 0.88), (xo + 0.17, 0.80, 0.88), (xo + 0.17, 0.87, 0.86),
               (xo - 0.17, 0.87, 0.86), (22, 22, 24), n=(0, -1, 0), double=True)
        m.quad((xo - 0.03, 0.82, 0.9), (xo + 0.03, 0.82, 0.9), (xo + 0.03, 1.02, 0.96),
               (xo - 0.03, 1.02, 0.96), (30, 30, 33), n=(1, 0, 0), double=True)
    # window posts (slim)
    m.quad((-1.40, -0.62, 1.05), (-1.34, -0.62, 1.05), (-1.28, 0.2, 1.18), (-1.34, 0.2, 1.18),
           frame, n=(0, 0, -1), double=True)
    m.quad((1.34, -0.62, 1.05), (1.40, -0.62, 1.05), (1.34, 0.2, 1.18), (1.28, 0.2, 1.18),
           frame, n=(0, 0, -1), double=True)
    # centre post offset to the right of the captain's sight line
    m.quad((0.26, -0.66, 1.02), (0.33, -0.66, 1.02), (0.32, 0.1, 1.2), (0.25, 0.1, 1.2),
           frame, n=(0, 0, -1), double=True)
    # instrument screens (faint glow)
    for xo in (-0.78, -0.26, 0.26, 0.78):
        m.quad((xo - 0.17, 0.54, 1.15), (xo + 0.17, 0.54, 1.15), (xo + 0.17, 0.82, 1.06),
               (xo - 0.17, 0.82, 1.06), (16, 28, 32), n=(0, -0.3, -1), mat=MAT_UNLIT)
    return m.finalize()


def cloud_mesh(rng, n=24, area=24000.0, alt=(1700.0, 4200.0)):
    m = Mesh()
    for _ in range(n):
        cx = rng.uniform(1, -area, area)
        cy = rng.uniform(1, -area, area)
        cz = -rng.uniform(1, *alt)
        for _ in range(int(rng.uniform(1, 2, 5))):
            px = cx + rng.uniform(1, -900, 900)
            py = cy + rng.uniform(1, -700, 700)
            pr = rng.uniform(1, 380, 1150)
            zz = cz + rng.uniform(1, -70, 70)
            ang = np.linspace(0, 2 * np.pi, 9, endpoint=False) + rng.uniform(1, 0, 6)
            rad = pr * (0.7 + 0.5 * rng.rng.random(9))
            pts = [(px + np.cos(a) * r, py + np.sin(a) * r * 0.8, zz) for a, r in zip(ang, rad)]
            sh = int(232 - 34 * rng.rng.random())
            col = (sh, sh, min(255, sh + 8))
            for i in range(1, len(pts) - 1):
                m.tri(pts[0], pts[i], pts[i + 1], col, n=UP, double=True,
                        mat=MAT_UNLIT)
    return m.finalize()
