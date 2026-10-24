"""Procedural world terrain: a 102 km heightfield with a mip pyramid, baked
surface colours, and a distance-adaptive LOD ring mesh builder.

World frame is NED (X = north, Y = east, Z = down), so altitude h maps to
Z = -h.  Everything is metres.
"""
import numpy as np

from .mathx import ValueNoise, smoothstep

CELL = 100.0          # base grid resolution (m)
GRID = 1024           # base grid cells per side
EXTENT = CELL * GRID  # 102.4 km
ORIGIN = -EXTENT * 0.5
LEVELS = 6            # mip levels: 100, 200, 400, 800, 1600, 3200 m cells

# Ring configuration: (min chebyshev index, max chebyshev index) per LOD level.
# Inner edge of ring k overlaps the outer edge of ring k-1 so there are no holes;
# coarser levels get a small depth bias so they never z-fight the finer ones.
# ring spans live in Terrain.spans (adaptive, reach based)

AIRPORT_ELEV = 26.0
AIRPORT = np.array((0.0, 0.0))          # (north, east) of the airfield reference
RWY_LEN = 3200.0
RWY_WID = 45.0
CITY = np.array((-9500.0, -5200.0))     # downtown
LAKE = np.array((-16000.0, -13000.0))
SEA_LEVEL = 0.0

MAT_LAND = 0
MAT_WATER = 1
MAT_UNLIT = 2


def _disk(nx, ny, cx, cy, r):
    """1.0 inside the disc, smooth ramp to 0 outside (vectorised)."""
    d = np.sqrt((nx - cx) ** 2 + (ny - cy) ** 2)
    return 1.0 - smoothstep((d - r * 0.55) / (r * 0.75))


def _blur2(a):
    """3x3 binomial blur (1 2 1 / 2 4 2 / 1 2 1)/16, edge padded."""
    if a.ndim == 3:
        p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    else:
        p = np.pad(a, 1, mode="edge")
    c = p[1:-1, 1:-1]
    return (p[:-2, :-2] + p[2:, :-2] + p[:-2, 2:] + p[2:, 2:]
            + 2.0 * (p[1:-1, :-2] + p[1:-1, 2:] + p[:-2, 1:-1] + p[2:, 1:-1])
            + 4.0 * c) / 16.0


def _blur2c(a):
    return _blur2(a.astype(np.float32))


class Terrain:
    def __init__(self, seed=20260):
        self.cell = CELL
        self.grid = GRID
        self.x0 = ORIGIN
        n = GRID
        # grid coordinates (metres) of every vertex
        idx = np.arange(n + 1, dtype=np.float64)
        X = idx[:, None] * CELL + ORIGIN
        Y = idx[None, :] * CELL + ORIGIN
        XX = np.broadcast_to(X, (n + 1, n + 1))
        YY = np.broadcast_to(Y, (n + 1, n + 1))

        h = self._generate(XX, YY, seed)
        self.h0 = np.ascontiguousarray(h.astype(np.float32))
        self.c0 = self._colorise(XX, YY, self.h0, seed)

        # mip pyramid: symmetric blur + decimate so that level-k index i maps
        # exactly onto base index i*2^k (keeps world alignment).
        self.h = [self.h0]
        self.c = [self.c0]
        cur_h, cur_c = self.h0, self.c0
        for _ in range(LEVELS - 1):
            cur_h = np.ascontiguousarray(
                _blur2(cur_h)[::2, ::2].astype(np.float32))
            cur_c = np.ascontiguousarray(_blur2c(cur_c)[::2, ::2].astype(np.uint8))
            self.h.append(cur_h)
            self.c.append(cur_c)

        self.max_level = LEVELS - 1
        self.spans = (2, 3, 3, 3, 3, 3)
        self.reach = 0.0

    # ------------------------------------------------------------ generation
    def _generate(self, X, Y, seed):
        nz = ValueNoise(seed=seed, period=512)
        nz2 = ValueNoise(seed=seed + 991, period=256)

        s = 9000.0
        base = nz.fbm(X / s, Y / s, octaves=5)            # 0..1 rolling land
        detail = nz2.fbm(X / 900.0, Y / 900.0, octaves=4)
        hills = np.clip(base - 0.42, 0, 1) * 2.4
        h = AIRPORT_ELEV + hills * 420.0 + (detail - 0.5) * 46.0

        # ---- mountains to the north / north-west
        mt = smoothstep((X - 11000.0) / 12000.0) * (1.0 - 0.55 * smoothstep((-Y - 20000.0) / 9000.0))
        ridge = nz.ridged(X / 5200.0 + 11.3, Y / 5200.0 - 4.7, octaves=5)
        h += mt * (ridge ** 1.7) * 2650.0
        # secondary range to the far west
        mt2 = smoothstep((-Y - 15000.0) / 14000.0) * smoothstep((X + 4000.0) / 16000.0)
        h += mt2 * nz.ridged(X / 6100.0 - 3.1, Y / 6100.0 + 8.2, octaves=4) * 1250.0

        # ---- ocean to the east (approaches over water)
        coast = 8500.0 + nz2.fbm(X / 4200.0, Y / 4200.0, octaves=3) * 5200.0
        sea = smoothstep((Y - coast) / 7000.0)
        h = h * (1.0 - sea) + (-160.0 - 900.0 * nz.fbm(X / 12000.0, Y / 12000.0, octaves=3)) * sea
        # shelf / gentle slope near the shoreline
        h += sea * (1.0 - sea) * 24.0

        # ---- inland lake
        lk = _disk(X, Y, LAKE[0], LAKE[1], 6200.0)
        h = h * (1.0 - lk) + (-26.0) * lk

        # ---- flatten the airfield and its approaches
        af = _disk(X, Y, AIRPORT[0], AIRPORT[1], 9000.0)
        h = h * (1.0 - af) + AIRPORT_ELEV * af

        # ---- flatten the city bowl
        ct = _disk(X, Y, CITY[0], CITY[1], 8500.0)
        h = h * (1.0 - ct) + (17.0 + 1.0 * detail) * ct

        h = np.maximum(h, -1400.0)
        return h

    def _colorise(self, X, Y, h, seed):
        nz = ValueNoise(seed=seed + 55, period=256)
        n = h.shape[0]
        # slope from the height grid
        gx = np.zeros_like(h)
        gy = np.zeros_like(h)
        gx[1:-1] = (h[2:] - h[:-2]) * 0.5 / CELL
        gy[:, 1:-1] = (h[:, 2:] - h[:, :-2]) * 0.5 / CELL
        gx[0] = gx[1]
        gx[-1] = gx[-2]
        gy[:, 0] = gy[:, 1]
        gy[:, -1] = gy[:, -2]
        slope = np.sqrt(gx * gx + gy * gy)

        patch = nz.fbm(X / 380.0, Y / 380.0, octaves=3)
        patch2 = nz.fbm(X / 90.0 + 5.0, Y / 90.0 - 3.0, octaves=2)

        col = np.zeros(h.shape + (3,), dtype=np.float32)

        def put(rgb, w):
            nonlocal col
            w = np.clip(w, 0.0, 1.0)[..., None].astype(np.float32)
            col = col * (1.0 - w) + np.array(rgb, dtype=np.float32) * w

        # deep water -> shallow water
        deep = np.clip((-h - 6.0) / 220.0, 0, 1)
        put((26, 62, 96), np.clip(-h + 0.5, 0, 1))
        put((14, 38, 68), deep)
        put((58, 118, 132), np.clip((h + 2.5) / 4.0, 0, 1) * np.clip(-h + 3.0, 0, 1))
        # beach
        put((196, 182, 148), np.clip(1.0 - np.abs(h - 3.0) / 5.0, 0, 1) * 0.9)
        # lowland grass / scrub
        grass = np.clip((h - 2.0) / 26.0, 0, 1) * np.clip((150.0 - h) / 120.0, 0, 1)
        put((86, 116, 58), grass)
        put((104, 132, 66), grass * np.clip(patch * 1.6 - 0.35, 0, 1))
        put((72, 100, 52), grass * np.clip(0.9 - patch * 1.4, 0, 1))
        # farmland patchwork around the airfield
        farm = _disk(X, Y, 1500.0, -6500.0, 11000.0) * np.clip((h - 4.0) / 30.0, 0, 1) * np.clip((120.0 - h) / 90.0, 0, 1)
        field = (np.floor(X / 460.0) * 7.0 + np.floor(Y / 460.0) * 13.0) % 5.0
        put((128, 138, 66), farm * (field < 1))
        put((150, 148, 84), farm * ((field >= 1) & (field < 2)))
        put((96, 122, 60), farm * ((field >= 2) & (field < 3)))
        put((140, 122, 78), farm * (field >= 3))
        # forest
        forest = np.clip((h - 90.0) / 260.0, 0, 1) * np.clip((1250.0 - h) / 300.0, 0, 1)
        forest *= np.clip(patch * 2.2 - 0.55, 0, 1)
        put((46, 76, 44), forest)
        put((38, 64, 40), forest * np.clip(patch2 * 1.5 - 0.2, 0, 1))
        # rock / scree on steep slopes and high ground
        rock = np.clip((slope - 0.22) / 0.45, 0, 1) + np.clip((h - 700.0) / 700.0, 0, 1)
        rock = np.clip(rock, 0, 1) * np.clip(-h + 4.0, 0, 1)
        put((116, 108, 100), rock * 0.95)
        put((92, 86, 82), rock * np.clip(patch2 * 1.4 - 0.3, 0, 1))
        # snow caps
        snow = np.clip((h - 1450.0) / 420.0, 0, 1) * np.clip(1.25 - slope * 1.6, 0, 1)
        put((232, 236, 240), snow)
        # urban ground
        urban = _disk(X, Y, CITY[0], CITY[1], 7200.0) * np.clip((40.0 - h) / 25.0, 0, 1)
        put((96, 96, 94), urban * 0.92)
        put((72, 84, 62), urban * np.clip(patch2 * 1.5 - 0.6, 0, 1) * 0.7)
        # airfield mown grass
        af = _disk(X, Y, AIRPORT[0], AIRPORT[1], 6200.0) * np.clip((60.0 - h) / 30.0, 0, 1)
        put((104, 128, 72), af)
        strip = (np.floor(Y / 130.0) % 2.0)
        put((96, 120, 66), af * strip * 0.6)

        col = np.clip(col, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(col)

    # -------------------------------------------------------------- sampling
    def world_to_index(self, x, y, level=0):
        c = CELL * (1 << level)
        return (x - ORIGIN) / c, (y - ORIGIN) / c

    def height(self, x, y, level=0):
        """Bilinear height sample at arbitrary world coordinates."""
        h = self.h[level]
        n = h.shape[0] - 1
        fi, fj = self.world_to_index(x, y, level)
        i0 = np.clip(np.floor(fi).astype(np.int64), 0, n - 1)
        j0 = np.clip(np.floor(fj).astype(np.int64), 0, n - 1)
        ti = np.clip(fi - i0, 0, 1)
        tj = np.clip(fj - j0, 0, 1)
        h00 = h[i0, j0]
        h10 = h[i0 + 1, j0]
        h01 = h[i0, j0 + 1]
        h11 = h[i0 + 1, j0 + 1]
        a = h00 + (h10 - h00) * ti
        b = h01 + (h11 - h01) * ti
        return a + (b - a) * tj

    def color_at(self, x, y, level=2):
        c = self.c[level]
        n = c.shape[0] - 1
        fi, fj = self.world_to_index(x, y, level)
        i = np.clip(np.round(fi).astype(np.int64), 0, n)
        j = np.clip(np.round(fj).astype(np.int64), 0, n)
        return c[i, j]

    # ------------------------------------------------------------- LOD rings
    def lod_plan(self, agl, pitch_deg=-7.0, quality=1.0):
        """List of (level, a_index, b_index, radius_m) rings + inner cap.

        Cell size follows slant range so every ring subtends a roughly
        constant angle; a per-ring triangle budget forces coarser mips when a
        ring would get too expensive.
        """
        agl = max(0.0, agl)
        d_near = agl / max(0.20, np.tan(np.radians(abs(pitch_deg) + 21.0)))
        max_reach = float(np.clip(5000.0 + 6.0 * agl, 12000.0, 40000.0))
        budget = int(230 * quality)
        cap_b = int(np.sqrt(max(budget, 16)) * 0.5)
        # inner cap: coarse disc so nothing is missing under the aircraft
        lv = 0
        for lv in range(self.max_level + 1):
            c = CELL * (1 << lv)
            if round(max(d_near, 300.0) * 0.55 / c) <= cap_b:
                break
        c = CELL * (1 << lv)
        b = max(1, int(round(max(d_near, 300.0) * 0.55 / c)))
        rings = [(lv, 0, b)]
        r = (b + 0.5) * c
        span_base = 3
        for _ in range(10):
            if r > max_reach:
                break
            slant = np.hypot(r, agl)
            c_req = 0.062 * slant if slant < 8000 else 496.0 + 0.105 * (slant - 8000)
            lv = int(round(np.log2(np.clip(c_req, CELL, CELL * (1 << self.max_level)) / CELL)))
            lv = int(np.clip(lv, 0, self.max_level))
            for _try in range(6):
                c = CELL * (1 << lv)
                a = max(1, int(np.floor(r / c + 0.5)))
                b = a + span_base + (1 if lv >= 4 else 0)
                cells = (2 * b + 1) ** 2 - (2 * a - 1) ** 2
                if cells <= budget or lv >= self.max_level:
                    break
                lv += 1
            rings.append((lv, a, b))
            r = (b + 0.5) * c
        self.reach = r
        return rings

    def ring_mesh(self, cam_x, cam_y, base_level, quality=1.0, cam_agl=200.0,
                  cam_pitch=-7.0, rings=None):
        """Build the terrain mesh around a camera position.

        Returns (verts (V,3) NED, tris (T,3) int32, colors (T,3) float32,
        mat (T,) uint8, bias (T,) float32).
        """
        if rings is None:
            rings = self.lod_plan(cam_agl, cam_pitch, quality)
        all_v, all_t, all_c, all_m, all_b = [], [], [], [], []
        voff = 0
        for ring_i, (lv, a_ring, b_ring) in enumerate(rings):
            c = CELL * (1 << lv)
            hm = self.h[lv]
            cm = self.c[lv]
            nlim = hm.shape[0] - 1
            if a_ring > nlim - 1:
                break
            b_ring = min(b_ring, nlim)
            ci = int(round((cam_x - ORIGIN) / c))
            cj = int(round((cam_y - ORIGIN) / c))
            B = b_ring
            gi = np.arange(ci - B, ci + B + 2)
            gj = np.arange(cj - B, cj + B + 2)
            W = gi.size
            GI, GJ = np.meshgrid(gi, gj, indexing="ij")
            GIc = np.clip(GI, 0, nlim)
            GJc = np.clip(GJ, 0, nlim)
            H = hm[GIc, GJc]
            C = cm[GIc, GJc].astype(np.float32)
            Xw = GI.astype(np.float32) * c + ORIGIN
            Yw = GJ.astype(np.float32) * c + ORIGIN
            verts = np.stack((Xw.ravel(), Yw.ravel(), (-H).ravel()), axis=1)
            li = np.arange(2 * B + 1)
            LJ, LI = np.meshgrid(li, li, indexing="ij")
            dist = np.maximum(np.abs(LI - B), np.abs(LJ - B))
            keep = (dist >= a_ring) & (dist <= b_ring)
            LI = LI[keep]
            LJ = LJ[keep]
            if LI.size == 0:
                continue
            v00 = LI * W + LJ
            v10 = (LI + 1) * W + LJ
            v01 = LI * W + (LJ + 1)
            v11 = (LI + 1) * W + (LJ + 1)
            t1 = np.stack((v00, v01, v11), axis=1) + voff
            t2 = np.stack((v00, v11, v10), axis=1) + voff
            tris = np.concatenate((t1, t2), axis=0).astype(np.int32)
            h00 = H[LI, LJ]
            h11 = H[LI + 1, LJ + 1]
            h01 = H[LI, LJ + 1]
            h10 = H[LI + 1, LJ]
            tri_h = np.concatenate((h00 + h01 + h11, h00 + h11 + h10)) / 3.0
            col = np.concatenate((
                (C[LI, LJ] + C[LI, LJ + 1] + C[LI + 1, LJ + 1]) / 3.0,
                (C[LI, LJ] + C[LI + 1, LJ + 1] + C[LI + 1, LJ]) / 3.0,
            ), axis=0)
            mat = np.where(tri_h < 0.9, MAT_WATER, MAT_LAND).astype(np.uint8)
            bias = np.full(tris.shape[0], 0.03 + ring_i * 0.18, dtype=np.float32)
            all_v.append(verts.astype(np.float32))
            all_t.append(tris)
            all_c.append(col)
            all_m.append(mat)
            all_b.append(bias)
            voff += verts.shape[0]
        if not all_t:
            e = np.zeros((0, 3), np.float32)
            return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32),
                    e.copy(), np.zeros(0, np.uint8), np.zeros(0, np.float32))
        return (np.concatenate(all_v), np.concatenate(all_t),
                np.concatenate(all_c), np.concatenate(all_m),
                np.concatenate(all_b))


if __name__ == "__main__":
    import time
    t0 = time.time()
    tr = Terrain()
    print("terrain built in %.2f s" % (time.time() - t0))
    t0 = time.time()
    for _ in range(20):
        v, t, c, m, b = tr.ring_mesh(1234.0, -5678.0, tr.base_level(400.0))
    dt = (time.time() - t0) / 20 * 1000
    print("ring mesh: %d verts, %d tris, %.2f ms" % (v.shape[0], t.shape[0], dt))
    print("height at airport %.1f m, at city %.1f m" % (tr.height(0, 0), tr.height(*CITY)))
    print("ocean sample %.1f m" % tr.height(0, 30000))
    print("mountain sample %.1f m" % tr.height(30000, -5000))
