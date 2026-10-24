"""Small vector / quaternion / noise helpers.

World frame is NED:  X = north, Y = east, Z = down  (all metres).
Body frame is the usual aerospace one: x = forward, y = right wing, z = down.
Quaternions are stored as (w, x, y, z).
"""
import numpy as np

DEG = np.pi / 180.0
RAD = 180.0 / np.pi

# ----------------------------------------------------------------- quaternions


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dtype=np.float64)


def quat_norm(q):
    n = np.sqrt(q @ q)
    return q / n if n > 1e-12 else np.array((1.0, 0.0, 0.0, 0.0))


def quat_integrate(q, pqr, dt):
    """Advance attitude quaternion by body rates (rad/s) over dt."""
    w = np.array((0.0, pqr[0], pqr[1], pqr[2]))
    q = q + 0.5 * dt * quat_mul(q, w)
    return quat_norm(q)


def dcm(q):
    """Body -> world (NED) rotation matrix."""
    w, x, y, z = q
    return np.array((
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)),
        (2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)),
    ), dtype=np.float64)


def euler_to_quat(roll, pitch, yaw):
    """3-2-1 (yaw, pitch, roll) -> quaternion."""
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ), dtype=np.float64)


def quat_to_euler(q):
    """-> (roll, pitch, yaw) in radians."""
    R = dcm(q)
    sp = np.clip(-R[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sp)
    if abs(sp) > 0.99999:                      # near the gimbal singularity
        roll = np.arctan2(R[0, 1], R[0, 2])
        yaw = 0.0
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def wrap360(a):
    return a % 360.0


def heading_deg(q):
    return wrap360(quat_to_euler(q)[2] * RAD)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------- noise


class ValueNoise:
    """Tileable 2-D value noise with fBm / ridged helpers (vectorised)."""

    def __init__(self, seed=1337, period=512):
        rng = np.random.default_rng(seed)
        self.p = int(period)
        self.m = self.p - 1
        self.g = rng.random((self.p, self.p), dtype=np.float32)

    def _smooth(self, t):
        return t * t * (3.0 - 2.0 * t)

    def sample(self, x, y):
        xi = np.floor(x).astype(np.int64)
        yi = np.floor(y).astype(np.int64)
        tx = self._smooth(x - xi)
        ty = self._smooth(y - yi)
        xi &= self.m
        yi &= self.m
        x1 = (xi + 1) & self.m
        y1 = (yi + 1) & self.m
        g = self.g
        v00 = g[xi, yi]
        v10 = g[x1, yi]
        v01 = g[xi, y1]
        v11 = g[x1, y1]
        a = v00 + (v10 - v00) * tx
        b = v01 + (v11 - v01) * tx
        return a + (b - a) * ty

    def fbm(self, x, y, octaves=5, lac=2.03, gain=0.5):
        out = np.zeros(np.shape(x), dtype=np.float32)
        amp = 1.0
        tot = 0.0
        fx, fy = x, y
        for _ in range(octaves):
            out += amp * self.sample(fx, fy)
            tot += amp
            amp *= gain
            fx *= lac
            fy *= lac
        return out / tot

    def ridged(self, x, y, octaves=4, lac=2.07, gain=0.5):
        out = np.zeros(np.shape(x), dtype=np.float32)
        amp = 1.0
        tot = 0.0
        fx, fy = x, y
        for _ in range(octaves):
            n = 1.0 - np.abs(self.sample(fx, fy) * 2.0 - 1.0)
            out += amp * n * n
            tot += amp
            amp *= gain
            fx *= lac
            fy *= lac
        return out / tot


class Rng:
    """Deterministic helper for scattered world decoration."""

    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)

    def uniform(self, n, lo, hi):
        out = self.rng.uniform(lo, hi, n).astype(np.float64)
        return float(out[0]) if n == 1 else out
