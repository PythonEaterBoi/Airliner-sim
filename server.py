"""Headless streaming server.

Runs the simulation + the numpy software renderer in this process and streams
JPEG frames over a WebSocket to any phone / browser, which sends back touch
controls.  Pure asyncio + stdlib sockets (no web framework); numpy/Pillow (or
cv2) do the pixels.

    python3 -m sim.server --port 8000
"""
import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import struct
import time

import numpy as np

from .engine import Sim

try:
    import cv2
except Exception:
    cv2 = None
if cv2 is None:
    from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "static")
WS_KEY = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

RES_STEPS = ((320, 180), (384, 216), (448, 252), (512, 288), (640, 360))


class Inputs:
    def __init__(self):
        self.stick = (0.0, 0.0)
        self.pedals = 0.0
        self.throttle = None
        self.brakes = 0.0


class Client:
    def __init__(self, writer):
        self.writer = writer
        self.alive = True
        self.slow = 0
        self.last_ack = time.time()


def ws_frame(payload, opcode=2):
    out = bytearray()
    out.append(0x80 | opcode)
    n = len(payload)
    if n < 126:
        out.append(n)
    elif n < 65536:
        out.append(126)
        out.extend(struct.pack(">H", n))
    else:
        out.append(127)
        out.extend(struct.pack(">Q", n))
    out.extend(payload)
    return bytes(out)


async def read_ws_frame(reader):
    hdr = await reader.readexactly(2)
    op = hdr[0] & 0x0F
    masked = hdr[1] & 0x80
    n = hdr[1] & 0x7F
    if n == 126:
        n = struct.unpack(">H", await reader.readexactly(2))[0]
    elif n == 127:
        n = struct.unpack(">Q", await reader.readexactly(8))[0]
    key = await reader.readexactly(4) if masked else None
    data = await reader.readexactly(n)
    if key:
        data = bytes(b ^ key[i % 4] for i, b in enumerate(data))
    return op, data


class Server:
    def __init__(self, port, quality=1.0):
        self.port = port
        self.sim = Sim(quality=quality, w=448, h=252)
        self.clients = []
        self.inputs = Inputs()
        self.res_idx = 2
        self.frame_ema = 30.0
        self.running = True
        self._last_state = 0.0

    # ------------------------------------------------------------------ http
    async def handle(self, reader, writer):
        try:
            line = await asyncio.wait_for(reader.readline(), 15)
            if not line:
                writer.close()
                return
            parts = line.decode("latin1").split()
            if len(parts) < 2:
                writer.close()
                return
            method, path = parts[0], parts[1]
            headers = {}
            while True:
                hl = await asyncio.wait_for(reader.readline(), 15)
                if hl in (b"\r\n", b"\n", b""):
                    break
                k, _, v = hl.decode("latin1").partition(":")
                headers[k.strip().lower()] = v.strip()
            if "upgrade" in headers.get("upgrade", "").lower() or \
               "websocket" in headers.get("upgrade", "").lower():
                await self._ws_handshake(reader, writer, headers)
                return
            await self._http(writer, path)
        except Exception:
            try:
                writer.close()
            except Exception:
                pass

    async def _http(self, writer, path):
        if path in ("/", "/index.html"):
            fp = os.path.join(STATIC, "index.html")
            ctype = "text/html; charset=utf-8"
        else:
            fp = os.path.join(STATIC, path.lstrip("/").replace("..", ""))
            ctype = ("text/css" if fp.endswith(".css") else
                     "application/javascript" if fp.endswith(".js") else
                     "image/png" if fp.endswith(".png") else
                     "text/plain")
        try:
            with open(fp, "rb") as f:
                body = f.read()
        except OSError:
            body = b"404"
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 3\r\n"
                         b"Connection: close\r\n\r\n" + body)
            await writer.drain()
            writer.close()
            return
        writer.write(("HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %d\r\n"
                      "Cache-Control: no-store\r\nConnection: close\r\n\r\n"
                      % (ctype, len(body))).encode())
        writer.write(body)
        await writer.drain()
        writer.close()

    async def _ws_handshake(self, reader, writer, headers):
        key = headers.get("sec-websocket-key", "")
        acc = base64.b64encode(hashlib.sha1((key + WS_KEY).encode()).digest()).decode()
        writer.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                      "Connection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n" % acc).encode())
        await writer.drain()
        cl = Client(writer)
        self.clients.append(cl)
        try:
            while cl.alive:
                op, data = await asyncio.wait_for(read_ws_frame(reader), 30)
                if op == 8:
                    break
                if op == 9:
                    writer.write(ws_frame(data, 10))
                    await writer.drain()
                    continue
                if op == 1:
                    self._on_msg(data.decode("utf-8", "replace"))
        except Exception:
            pass
        finally:
            cl.alive = False
            if cl in self.clients:
                self.clients.remove(cl)
            try:
                writer.close()
            except Exception:
                pass

    # -------------------------------------------------------------- controls
    def _on_msg(self, txt):
        try:
            m = json.loads(txt)
        except Exception:
            return
        t = m.get("t")
        ac = self.sim.ac
        ap = ac.ap
        if t == "in":
            st = m.get("st")
            if st:
                self.inputs.stick = (float(np.clip(st[0], -1, 1)), float(np.clip(st[1], -1, 1)))
            if "pd" in m:
                self.inputs.pedals = float(np.clip(m["pd"], -1, 1))
            if "th" in m and m["th"] is not None:
                self.inputs.throttle = float(np.clip(m["th"], -0.45, 1.0))
            if "brk" in m:
                self.inputs.brakes = float(np.clip(m["brk"], 0, 1))
        elif t == "cmd":
            k = m.get("k")
            v = m.get("v")
            if k == "flapup":
                ac.flap_detent = max(0, ac.flap_detent - 1)
            elif k == "flapdn":
                ac.flap_detent = min(len(ac.FLAP_SET) - 1, ac.flap_detent + 1)
            elif k == "gear":
                ac.gear_cmd = 0 if ac.gear_cmd else 1
            elif k == "spd":
                ac.spoiler_cmd = 0.0 if ac.spoiler_cmd > 0.5 else 1.0
            elif k == "park":
                ac.park_brake = 0.0 if ac.park_brake > 0.5 else 1.0
            elif k == "ap":
                ap.toggle()
            elif k == "at":
                ap.toggle_at()
            elif k == "app":
                ap.arm_app()
            elif k == "cam":
                cams = ("cockpit", "chase", "wing", "tower")
                self.sim.camera = cams[(cams.index(self.sim.camera) + 1) % len(cams)]
            elif k == "pos":
                x, y, alt = (float(q) for q in v)
                sp = max(float(np.linalg.norm(ac.vel)), 70.0)
                ac.pos = np.array((x, y, -alt))
                ac.vel = np.array((sp, 0.0, 0.0))     # body axes: straight ahead
                self.sim.chase_pos = None
            elif k == "sit":
                ac.reset_situation(v or "approach")
                self.sim.chase_pos = None
            elif k == "tod":
                self.sim.set_time(float(v))
            elif k == "trim":
                ac.trim = float(np.clip(ac.trim + float(v), -0.32, 0.32))
            elif k == "q":
                self.res_idx = int(np.clip(self.res_idx + int(v), 0, len(RES_STEPS) - 1))
                self.sim.set_resolution(*RES_STEPS[self.res_idx])
            elif k == "qual":
                self.sim.set_quality(float(v))
            elif k == "hdg":
                ap.t_hdg = float(v)
            elif k == "alt":
                ap.t_alt = float(v)
            elif k == "vs":
                ap.t_vs = float(v)
                ap.pitch_mode = "VS"
            elif k == "ias":
                ap.t_ias = float(v)
            elif k == "pm":
                ap.pitch_mode = v
            elif k == "rm":
                ap.roll_mode = v

    # ------------------------------------------------------------------ loop
    def _encode(self, fb):
        if cv2 is not None:
            ok, buf = cv2.imencode(".jpg", fb[:, :, ::-1],
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            return buf.tobytes() if ok else b""
        b = io.BytesIO()
        Image.fromarray(fb).save(b, "JPEG", quality=72, optimize=False)
        return b.getvalue()

    async def loop(self):
        loop = asyncio.get_event_loop()
        last = time.time()
        st_every = 0
        while self.running:
            t0 = time.time()
            dt = min(max(t0 - last, 1 / 240.0), 1 / 12.0)
            last = t0
            ac = self.sim.ac
            tgt = np.array(self.inputs.stick, float)
            tgt = tgt * 0.55 + 0.45 * tgt * np.abs(tgt)   # soften around centre
            ac.stick += np.clip(tgt - ac.stick, -3.0 * dt, 3.0 * dt)
            ac.pedals += float(np.clip(self.inputs.pedals - ac.pedals,
                                       -4.0 * dt, 4.0 * dt))
            ac.brake_cmd = self.inputs.brakes
            if self.inputs.throttle is not None:
                ac.throttle_cmd = self.inputs.throttle
            self.sim.step(dt)
            fb = await loop.run_in_executor(None, self.sim.render)
            self.frame_ema = 0.9 * self.frame_ema + 0.1 * self.sim.frame_ms
            self._adapt()
            jpg = self._encode(fb)
            msg = ws_frame(jpg, 2)
            dead = []
            st_now = time.time()
            send_state = (st_now - self._last_state) > 0.08
            if send_state:
                self._last_state = st_now
                stmsg = ws_frame(json.dumps(self.sim.state()).encode(), 1)
            for cl in list(self.clients):
                try:
                    tr = cl.writer.transport
                    if tr.get_write_buffer_size() > 250000:
                        cl.slow += 1          # client not reading: drop frames,
                        if cl.slow > 90:      # then give up on it
                            dead.append(cl)
                        continue
                    cl.slow = 0
                    cl.writer.write(msg)
                    if send_state:
                        cl.writer.write(stmsg)
                except Exception:
                    dead.append(cl)
            for cl in dead:
                cl.alive = False
                if cl in self.clients:
                    self.clients.remove(cl)
            self.sim.fps = 0.9 * self.sim.fps + 0.1 * (1.0 / max(dt, 1e-3))
            elapsed = time.time() - t0
            await asyncio.sleep(max(0.0, 1.0 / 30.0 - elapsed))

    def _adapt(self):
        if self.frame_ema > 46.0 and self.res_idx > 0:
            self.res_idx -= 1
            self.sim.set_resolution(*RES_STEPS[self.res_idx])
            self.frame_ema = 34.0
        elif self.frame_ema < 21.0 and self.res_idx < len(RES_STEPS) - 1:
            self.res_idx += 1
            self.sim.set_resolution(*RES_STEPS[self.res_idx])
            self.frame_ema = 30.0


async def main(port, quality):
    srv = Server(port, quality)
    server = await asyncio.start_server(srv.handle, "0.0.0.0", port)
    print("airliner sim streaming on 0.0.0.0:%d (build %.1fs)" % (port, srv.sim.build_time),
          flush=True)
    await asyncio.gather(server.serve_forever(), srv.loop())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--quality", type=float, default=1.0)
    a = ap.parse_args()
    asyncio.run(main(a.port, a.quality))
