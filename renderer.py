"""A small software 3-D renderer written on numpy.

Pipeline per frame:
  sky pass (half-res view-direction lookup, upscaled)
  -> meshes are transformed to view space, lit and fogged, and appended to a
     single triangle buffer
  -> vectorised near-plane clip (1 in-triangle can emit 2 out-triangles)
  -> perspective divide, back-face + viewport cull
  -> back-to-front sort, then z-buffered scan conversion
     (tiny triangles take a batched point-splat fast path)
  -> optional pre-baked cockpit overlay

Everything is CPU only, no GPU / no display needed, so frames can be rendered
head-less on a server and streamed to a phone.
"""
import numpy as np

try:                                   # cv2 is the fast path for resize / jpeg
    import cv2
except Exception:                      # pragma: no cover
    cv2 = None

MAT_LAND = 0
MAT_WATER = 1
MAT_UNLIT = 2


class Sky:
    """Lighting + haze model driven by a sun elevation angle."""

    def __init__(self):
        self.set(0.7)

    def set(self, sun_elev_deg, sun_azim_deg=115.0):
        e = np.radians(sun_elev_deg)
        a = np.radians(sun_azim_deg)
        # direction *towards* the sun, NED
        self.sun = np.array((np.cos(e) * np.cos(a), np.cos(e) * np.sin(a),
                             -np.sin(e)), dtype=np.float64)
        t = np.clip(np.sin(e) * 0.5 + 0.5, 0.0, 1.0)       # 0 night .. 1 noon
        low = np.clip(1.0 - abs(sun_elev_deg) / 22.0, 0.0, 1.0)
        zen = self._mix((8, 12, 30), (38, 92, 168), t)
        zen = self._mix(zen, (86, 74, 128), low * 0.55)
        hor = self._mix((24, 30, 52), (158, 190, 222), t)
        hor = self._mix(hor, (238, 152, 88), low * 0.85)
        self.zenith = zen
        self.horizon = hor
        self.ground_haze = self._mix((10, 12, 20), (126, 142, 154), t)
        self.sun_color = self._mix((90, 100, 150), (255, 250, 235), t)
        self.sun_color = self._mix(self.sun_color, (255, 168, 90), low * 0.8)
        self.diffuse = 0.28 + 0.86 * t
        self.ambient = 0.16 + 0.42 * t + 0.10 * low
        self.night = 1.0 - np.clip((sun_elev_deg + 6.0) / 12.0, 0.0, 1.0)
        self.key = (round(sun_elev_deg * 2), round(sun_azim_deg * 2))

    @staticmethod
    def _mix(a, b, t):
        return np.array(a, np.float32) * (1 - t) + np.array(b, np.float32) * t

    def gradient_lut(self, n=128):
        """Colour ramp indexed by (elevation+1)/2, 0 = straight down."""
        t = np.linspace(0.0, 1.0, n)[:, None]
        hor_i = n * 0.5
        lut = np.zeros((n, 3), np.float32)
        up = np.clip((t * n - hor_i) / (n - hor_i), 0, 1) ** 0.72
        lut[:] = self.horizon
        lut = lut * (1 - up) + self.zenith * up
        dn = np.clip((hor_i - t * n) / hor_i, 0, 1)
        lut = lut * (1 - dn) + self.ground_haze * dn
        return lut


class Renderer:
    MAXT = 60000

    def __init__(self, w=480, h=270, hfov=70.0):
        self.near = 0.35
        self.sky = Sky()
        self.cockpit_mask = None
        self.cockpit_col = None
        self._tv = np.zeros((self.MAXT, 3, 3), np.float32)
        self._tc = np.zeros((self.MAXT, 3), np.float32)
        self._tm = np.zeros(self.MAXT, np.uint8)
        self._tb = np.zeros(self.MAXT, np.float32)
        self._n = 0
        self.stats = {}
        self.resize(w, h, hfov)

    # ------------------------------------------------------------- setup
    def resize(self, w, h, hfov=None):
        self.w = int(w)
        self.h = int(h)
        if hfov:
            self.hfov = float(hfov)
        self.f = (self.w * 0.5) / np.tan(np.radians(self.hfov) * 0.5)
        self.cx = self.w * 0.5
        self.cy = self.h * 0.5
        self.fb = np.zeros((self.h, self.w, 3), np.uint8)
        self.zb = np.zeros((self.h, self.w), np.float32)
        self.cockpit_mask = None
        self._sky_lut = None
        self._sky_lut_key = None
        self._bigx = np.arange(129, dtype=np.float32)
        self._bigy = np.arange(129, dtype=np.float32)
        rr = np.arange(129, dtype=np.int32)
        self._rows = (rr[:, None] * np.int32(self.w)) * np.ones((1, 129), np.int32)
        self._cols = np.ones((129, 1), np.int32) * rr[None, :]

    def set_camera(self, pos, R):
        """pos: world NED; R: body->world DCM.  View axes = (right, down, fwd)."""
        self.cam_pos = np.asarray(pos, np.float64)
        self.R = np.asarray(R, np.float64)
        self.M = self.R[:, (1, 2, 0)].T          # rows: right, down, forward
        self.Mt = np.ascontiguousarray(self.M.T)

    # ------------------------------------------------------------- meshes
    def begin(self):
        self._n = 0
        self.zb.fill(np.inf)
        self._draw_sky()

    def add_mesh(self, verts, tris, colors, mat=None, bias=None,
                 rot=None, pos=None, fog_scale=1.0, shade=True):
        """verts (V,3) world or model space, tris (T,3), colors (T,3)."""
        nt = tris.shape[0]
        if nt == 0:
            return
        n0 = self._n
        if n0 + nt * 2 > self.MAXT:
            return
        v = np.asarray(verts, np.float32)
        if rot is not None:
            v = v @ np.asarray(rot, np.float32).T
        if pos is not None:
            v = v + np.asarray(pos, np.float32)
        t = np.asarray(tris, np.int64)
        p = (v - self.cam_pos.astype(np.float32)) @ self.Mt.astype(np.float32)
        tri = p[t]                                        # (T,3,3) view space
        base = np.asarray(colors, np.float32)
        if base.ndim == 1 or base.shape[0] != nt:         # per-vertex colours
            base = base[t].mean(axis=1)
        mm = np.zeros(nt, np.uint8) if mat is None else np.asarray(mat, np.uint8)
        col = self._shade(v, t, base, mm, tri) if shade else base.copy()
        depth = tri[:, :, 2].mean(axis=1)
        fog = self._fog(depth, fog_scale)
        col = col * (1.0 - fog)[:, None] + self.sky.horizon * fog[:, None]
        emis = mm == MAT_UNLIT
        if emis.any():   # lights keep glowing through haze
            col[emis] = np.maximum(col[emis], base[emis] *
                                   (0.45 + 0.55 * (1.0 - fog[emis]))[:, None])
        self._tv[n0:n0 + nt] = tri
        self._tc[n0:n0 + nt] = np.clip(col, 0, 255)
        self._tm[n0:n0 + nt] = mm
        if bias is None:
            self._tb[n0:n0 + nt] = 0.0
        elif np.isscalar(bias):
            self._tb[n0:n0 + nt] = float(bias)
        else:
            b = np.asarray(bias, np.float32)
            self._tb[n0:n0 + nt] = b if b.shape[0] == nt else float(b.mean())
        self._n = n0 + nt

    def add_view_mesh(self, tri, colors, mat=None):
        """Triangles already in view space (used for the cockpit shell)."""
        n0 = self._n
        nt = tri.shape[0]
        if n0 + nt * 2 > self.MAXT:
            return
        self._tv[n0:n0 + nt] = tri
        self._tc[n0:n0 + nt] = colors
        self._tm[n0:n0 + nt] = 0 if mat is None else mat
        self._tb[n0:n0 + nt] = 0.0
        self._n = n0 + nt

    # ------------------------------------------------------------- shading
    def _shade(self, v, t, col, mat, tri):
        e0 = v[t[:, 1]] - v[t[:, 0]]
        e1 = v[t[:, 2]] - v[t[:, 0]]
        n = np.cross(e0, e1)
        ln = np.sqrt(np.einsum("ij,ij->i", n, n))
        ln[ln < 1e-9] = 1e-9
        n /= ln[:, None]
        ndl = n @ self.sky.sun.astype(np.float32)
        lam = np.clip(ndl, 0.0, 1.0)
        amb = self.sky.ambient
        dif = self.sky.diffuse
        lit = amb + dif * lam
        # fake sky bounce so undersides are not black
        lit = lit + 0.16 * np.clip(-n[:, 2] * 0.5 + 0.5, 0, 1) * self.sky.ambient
        out = col * lit[:, None]
        w = mat == MAT_WATER
        if w.any():
            view = tri[w].mean(axis=1)
            vl = np.sqrt(np.einsum("ij,ij->i", view, view))
            vl[vl < 1e-9] = 1e-9
            view /= vl[:, None]
            nn = n[w]
            sunv = self.sky.sun.astype(np.float32)
            d = 2.0 * (nn @ sunv)[:, None] * nn - sunv
            spec = np.clip(np.einsum("ij,ij->i", d, view), 0.0, 1.0) ** 24.0
            fres = 0.10 + 0.35 * np.clip(1.0 - nn[:, 2], 0, 1) ** 2
            out[w] = np.clip(out[w] * (1.0 + fres[:, None]) +
                             (self.sky.sun_color * (spec * 0.9)[:, None]), 0, 255)
        u = mat == MAT_UNLIT
        if u.any():
            out[u] = col[u]
        return out

    def _fog(self, depth, scale=1.0):
        vis = self.fog_range * scale
        f = 1.0 - np.exp(-np.clip(depth, 0, 1e6) / vis)
        return np.clip(f * f * (3.0 - 2.0 * f), 0.0, 1.0).astype(np.float32)

    fog_range = 16000.0

    # ---------------------------------------------------------------- sky
    def _draw_sky(self):
        lut_key = (self.sky.key, self.w, self.h)
        if self._sky_lut is None or self._sky_lut_key != self.sky.key:
            self._sky_lut = self.sky.gradient_lut(128)
            self._sky_lut_key = self.sky.key
        lut = self._sky_lut
        W, H = self.w, self.h
        sw, sh = max(2, W // 2), max(2, H // 2)
        f = self.f * sw / W
        xs = (np.arange(sw, dtype=np.float32) + 0.5) * (W / sw) - self.cx
        ys = (np.arange(sh, dtype=np.float32) + 0.5) * (H / sh) - self.cy
        dx = (xs / f)[None, :, None]
        dy = (ys / f)[:, None, None]
        dv = np.concatenate((np.broadcast_to(dx, (sh, sw, 1)),
                             np.broadcast_to(dy, (sh, sw, 1)),
                             np.ones((sh, sw, 1), np.float32)), axis=2)
        dw = dv @ self.M.astype(np.float32)
        nrm = np.sqrt(np.einsum("ijk,ijk->ij", dw, dw))
        nrm[nrm < 1e-9] = 1e-9
        elev = -dw[:, :, 2] / nrm
        idx = np.clip(((elev + 1.0) * 63.5).astype(np.int32), 0, 127)
        col = lut[idx]
        sd = self.sky.sun.astype(np.float32)
        cs = np.einsum("ijk,k->ij", dw / nrm[:, :, None], sd)
        glow = np.clip(cs, 0, 1)
        col = col + self.sky.sun_color.astype(np.float32) * (
            (glow ** 220.0) * 1.25 + (glow ** 9.0) * 0.30 +
            (glow ** 2.2) * 0.10)[:, :, None]
        col = np.clip(col, 0, 255).astype(np.uint8)
        if cv2 is not None:
            big = cv2.resize(col, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            big = np.repeat(np.repeat(col, 2, 0), 2, 1)[:H, :W]
        self.fb[:] = big

    # ------------------------------------------------------------ draw
    def draw(self):
        n = self._n
        if n == 0:
            self.stats = {"tris": 0, "drawn": 0}
            return self.fb
        P = self._tv[:n]
        col = self._tc[:n]
        mat = self._tm[:n]
        zn = self.near
        s = P[:, :, 2] - zn
        ins = s >= 0.0
        nin = ins.sum(axis=1)
        part = (nin > 0) & (nin < 3)
        if part.any():
            k = np.where(nin == 1, np.argmax(ins, axis=1), np.argmax(~ins, axis=1))
            r = (k - 1) % 3
            jj = (np.arange(3)[None, :] + r[:, None]) % 3
            rows = np.arange(n)[:, None]
            Pc = P[rows, jj]
            sc = s[rows, jj]
            p0, p1, p2 = Pc[:, 0], Pc[:, 1], Pc[:, 2]
            s0, s1, s2 = sc[:, 0], sc[:, 1], sc[:, 2]
            d0 = s1 - s0
            d2 = s1 - s2
            d0 = np.where(np.abs(d0) < 1e-6, 1e-6, d0)
            d2 = np.where(np.abs(d2) < 1e-6, 1e-6, d2)
            A = p1 + (s1 / d0)[:, None] * (p0 - p1)
            B = p1 + (s1 / d2)[:, None] * (p2 - p1)
            one_in = (nin == 1)[:, None, None]
            all_in = (nin == 3)[:, None, None]
            out0 = np.where(all_in, P,
                            np.where(one_in, np.stack((A, p1, B), 1),
                                     np.stack((p0, A, B), 1)))
            out1 = np.stack((p0, B, p2), 1)
            v0 = nin > 0
            v1 = nin == 2
            P = np.concatenate((out0[v0], out1[v1]))
            col = np.concatenate((col[v0], col[v1]))
            mat = np.concatenate((mat[v0], mat[v1]))
            bias = np.concatenate((self._tb[:n][v0], self._tb[:n][v1]))
        else:
            v0 = nin == 3
            P = P[v0]
            mat = mat[v0]
            col = col[v0]
            bias = self._tb[:n][v0]
        if P.shape[0] == 0:
            self.stats = {"tris": n, "drawn": 0}
            return self.fb
        P = P.astype(np.float32, copy=False)
        z = P[:, :, 2]
        z = np.maximum(z, zn)
        inv = self.f / z
        sx = P[:, :, 0] * inv + self.cx
        sy = P[:, :, 1] * inv + self.cy
        W, H = self.w, self.h
        minx = sx.min(axis=1)
        maxx = sx.max(axis=1)
        miny = sy.min(axis=1)
        maxy = sy.max(axis=1)
        area = ((sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0]) -
                (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0]))
        vis = (maxx > -0.5) & (minx < W + 0.5) & (maxy > -0.5) & (miny < H + 0.5)
        # front faces are negative here; sub-pixel emissive triangles (lights)
        # survive the cull so the rasteriser's splat path can draw them as dots
        vis &= (area < -0.35) | ((mat == MAT_UNLIT) & (area < 0.0))
        if not vis.any():
            self.stats = {"tris": n, "drawn": 0}
            return self.fb
        P, sx, sy, col, area, bias = P[vis], sx[vis], sy[vis], col[vis], area[vis], bias[vis]
        zc = P[:, :, 2].mean(axis=1)
        order = np.argsort(-zc, kind="stable")
        sx, sy = sx[order], sy[order]
        zt = P[order][:, :, 2] + bias[order][:, None]
        col = col[order]
        area = area[order]
        self._rasterise(sx, sy, zt, col, area)
        self.stats = {"tris": n, "drawn": int(order.size)}
        return self.fb

    # --------------------------------------------------------- rasteriser
    def _rasterise(self, sx, sy, zt, col, area):
        W, H = self.w, self.h
        fbf = self.fb.reshape(-1, 3)
        zbf = self.zb.reshape(-1)
        fx0 = np.floor(sx.min(axis=1))
        fx1 = np.floor(sx.max(axis=1))
        fy0 = np.floor(sy.min(axis=1))
        fy1 = np.floor(sy.max(axis=1))
        ix0 = np.clip(fx0, 0, W - 1).astype(np.int32)
        ix1 = np.clip(fx1, 0, W - 1).astype(np.int32)
        iy0 = np.clip(fy0, 0, H - 1).astype(np.int32)
        iy1 = np.clip(fy1, 0, H - 1).astype(np.int32)
        bw = ix1 - ix0 + 1
        bh = iy1 - iy0 + 1
        n = sx.shape[0]
        tiny = (bw * bh) <= 2
        if tiny.any():
            self._splat(tiny, (fx0 + fx1 + 2.0) * 0.5, (fy0 + fy1 + 2.0) * 0.5,
                        zt.mean(axis=1), col)
        huge = (~tiny) & ((bw > 128) | (bh > 128))
        if huge.any():
            for i in np.nonzero(huge)[0]:
                self._raster_one(i, sx, sy, zt, col, ix0, ix1, iy0, iy1)
        rest = np.nonzero(~tiny & ~huge)[0]
        if rest.size == 0:
            return
        key = np.maximum(bw, bh)
        edges = (8, 16, 32, 64, 128)
        ks = (64, 48, 32, 20, 10)
        prev = np.zeros(rest.size, bool)
        batches = []
        for e, k in zip(edges, ks):
            sel = (key[rest] <= e) & ~prev
            prev |= sel
            ids = rest[sel]
            for s0 in range(0, ids.size, k):
                batches.append(ids[s0:s0 + k])
        a0 = sy[:, 1] - sy[:, 2]
        b0 = sx[:, 2] - sx[:, 1]
        c0 = sx[:, 1] * sy[:, 2] - sx[:, 2] * sy[:, 1]
        a1 = sy[:, 2] - sy[:, 0]
        b1 = sx[:, 0] - sx[:, 2]
        c1 = sx[:, 2] * sy[:, 0] - sx[:, 0] * sy[:, 2]
        a2 = sy[:, 0] - sy[:, 1]
        b2 = sx[:, 1] - sx[:, 0]
        c2 = sx[:, 0] * sy[:, 1] - sx[:, 1] * sy[:, 0]
        ia = 1.0 / area
        izt = 1.0 / np.maximum(zt, 1e-4)
        z0, z1, z2 = izt[:, 0], izt[:, 1], izt[:, 2]
        BIGX = self._bigx
        BIGY = self._bigy
        ROWS = self._rows
        COLS = self._cols
        stride = np.int32(W)
        for ids in batches:
            bwk = bw[ids]
            bhk = bh[ids]
            BW = int(bwk.max())
            BH = int(bhk.max())
            gx = BIGX[:BW] + 0.5
            gy = BIGY[:BH] + 0.5
            X = ix0[ids][:, None, None] + gx[None, None, :]
            Y = iy0[ids][:, None, None] + gy[None, :, None]
            w0 = a0[ids][:, None, None] * X + (b0[ids][:, None, None] * Y + c0[ids][:, None, None])
            w1 = a1[ids][:, None, None] * X + (b1[ids][:, None, None] * Y + c1[ids][:, None, None])
            w2 = a2[ids][:, None, None] * X + (b2[ids][:, None, None] * Y + c2[ids][:, None, None])
            wmax = np.maximum(np.maximum(w0, w1), w2)
            iw = (w0 * z0[ids][:, None, None] + w1 * z1[ids][:, None, None]
                  + w2 * z2[ids][:, None, None]) * ia[ids][:, None, None]
            zz = 1.0 / np.where(np.abs(iw) < 1e-9, 1e-9, iw)
            pad = ((gx[None, None, :] < (ix0[ids] + bwk)[:, None, None]) &
                   (gy[None, :, None] < (iy0[ids] + bhk)[:, None, None]))
            px = ((iy0[ids] * stride + ix0[ids])[:, None, None] +
                  ROWS[:BH, :BW][None] + COLS[:BH, :BW][None])
            np.clip(px, 0, W * H - 1, out=px)
            cur = zbf[px]
            ok = (wmax <= 0.0) & pad & (zz < cur)
            mo = ok.reshape(-1)
            if not mo.any():
                continue
            idx = px.reshape(-1)[mo]
            zbf[idx] = zz.reshape(-1)[mo]
            fbf[idx] = col[ids][np.flatnonzero(mo) // (BH * BW)]

    def _raster_one(self, i, sx, sy, zt, col, ix0, ix1, iy0, iy1):
        W = self.w
        bx0, bx1 = int(ix0[i]), int(ix1[i])
        by0, by1 = int(iy0[i]), int(iy1[i])
        gx = np.arange(bx0, bx1 + 1, dtype=np.float32) + 0.5
        gy = np.arange(by0, by1 + 1, dtype=np.float32) + 0.5
        w0 = (sy[i, 1] - sy[i, 2]) * gx[None, :] + ((sx[i, 2] - sx[i, 1]) * gy[:, None] +
                                                    sx[i, 1] * sy[i, 2] - sx[i, 2] * sy[i, 1])
        w1 = (sy[i, 2] - sy[i, 0]) * gx[None, :] + ((sx[i, 0] - sx[i, 2]) * gy[:, None] +
                                                    sx[i, 2] * sy[i, 0] - sx[i, 0] * sy[i, 2])
        w2 = (sy[i, 0] - sy[i, 1]) * gx[None, :] + ((sx[i, 1] - sx[i, 0]) * gy[:, None] +
                                                    sx[i, 0] * sy[i, 1] - sx[i, 1] * sy[i, 0])
        mask = np.maximum(np.maximum(w0, w1), w2) <= 0.0
        if not mask.any():
            return
        ar = ((sx[i, 1] - sx[i, 0]) * (sy[i, 2] - sy[i, 0]) -
              (sx[i, 2] - sx[i, 0]) * (sy[i, 1] - sy[i, 0]))
        iz = 1.0 / np.maximum(zt[i], 1e-4)
        zz = 1.0 / ((w0 * iz[0] + w1 * iz[1] + w2 * iz[2]) / ar)
        zsub = self.zb[by0:by1 + 1, bx0:bx1 + 1]
        ok = mask & (zz < zsub)
        if not ok.any():
            return
        zsub[ok] = zz[ok]
        self.fb[by0:by1 + 1, bx0:bx1 + 1][ok] = col[i]

    def _splat(self, sel, px, py, pz, col):
        """Depth-tested 2x2 glow dots for sub-pixel emissive triangles (lights)."""
        W, H = self.w, self.h
        ix = np.floor(px[sel]).astype(np.int32)
        iy = np.floor(py[sel]).astype(np.int32)
        zz = pz[sel] * 0.985          # slight bias: win against ground slivers
        cc = col[sel]
        for ox, oy, dim in ((0, 0, 1.0), (1, 0, 0.75), (0, 1, 0.75), (1, 1, 0.55)):
            jx, jy = ix + ox, iy + oy
            good = (jx >= 0) & (jx < W) & (jy >= 0) & (jy < H)
            jx, jy, z2, c2 = jx[good], jy[good], zz[good], cc[good] * dim
            if jx.size == 0:
                continue
            idx = jy * W + jx
            srt = np.argsort(idx, kind="stable")
            idx, z2, c2 = idx[srt], z2[srt], c2[srt]
            first = np.empty(idx.size, bool)
            first[0] = True
            np.not_equal(idx[1:], idx[:-1], out=first[1:])
            idx, z2, c2 = idx[first], z2[first], c2[first]
            iy2, ix2 = idx // W, idx % W
            cur = self.zb[iy2, ix2]
            ok = z2 < cur
            if ok.any():
                self.zb[iy2[ok], ix2[ok]] = z2[ok]
                self.fb[iy2[ok], ix2[ok]] = c2[ok]

    # ------------------------------------------------- cockpit overlay
    def bake_cockpit(self, verts, tris, colors):
        """Render the (view-space static) cockpit once into an overlay layer."""
        W, H = self.w, self.h
        fb = np.zeros((H, W, 3), np.uint8)
        zb = np.full((H, W), np.inf, np.float32)
        saved_fb, saved_zb = self.fb, self.zb
        self.fb, self.zb = fb, zb
        try:
            tri = np.asarray(verts, np.float32)[np.asarray(tris, np.int64)]
            z = np.maximum(tri[:, :, 2], 1e-3)
            inv = self.f / z
            sx = tri[:, :, 0] * inv + self.cx
            sy = tri[:, :, 1] * inv + self.cy
            area = ((sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0]) -
                    (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0]))
            keep = area < -0.35
            order = np.argsort(-tri[keep][:, :, 2].mean(axis=1), kind="stable")
            self._rasterise(sx[keep][order], sy[keep][order],
                            tri[keep][order][:, :, 2],
                            np.asarray(colors, np.float32)[keep][order],
                            area[keep][order])
        finally:
            self.fb, self.zb = saved_fb, saved_zb
        self.cockpit_mask = zb < np.inf
        self.cockpit_col = fb
        return self.cockpit_mask.sum()

    def apply_cockpit(self):
        if self.cockpit_mask is not None:
            m = self.cockpit_mask
            self.fb[m] = self.cockpit_col[m]
