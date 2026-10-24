"""WS client harness: grab frames / drive the sim like a phone would."""
import base64, hashlib, json, os, socket, struct, sys, time


class WSC:
    def __init__(self, port=8000):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=20)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall(("GET /ws HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
                        "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
                        "Sec-WebSocket-Version: 13\r\n\r\n" % key).encode())
        r = b""
        while b"\r\n\r\n" not in r:
            r += self.s.recv(4096)
        assert b"101" in r.split(b"\r\n")[0], r[:80]

    def _rx(self, n):
        d = b""
        while len(d) < n:
            c = self.s.recv(n - len(d))
            if not c:
                raise EOFError
            d += c
        return d

    def frame(self, timeout=6.0):
        t0 = time.time()
        self.s.settimeout(timeout)
        while time.time() - t0 < timeout:
            h = self._rx(2)
            op, n = h[0] & 0xF, h[1] & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._rx(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._rx(8))[0]
            data = self._rx(n)
            if op == 2:
                return data
            if op == 1:
                self.last_state = json.loads(data)
        return None

    def grab(self, secs=1.0):
        """drain for secs, return last jpeg"""
        t0 = time.time()
        jpg = None
        while time.time() - t0 < secs:
            j = self.frame(2)
            if j:
                jpg = j
        return jpg

    def send(self, obj):
        p = json.dumps(obj).encode()
        mask = os.urandom(4)
        hd = bytearray([0x81])
        n = len(p)
        if n < 126:
            hd.append(0x80 | n)
        elif n < 65536:
            hd.append(0x80 | 126)
            hd.extend(struct.pack(">H", n))
        hd.extend(mask)
        self.s.sendall(bytes(hd) + bytes(b ^ mask[i % 4] for i, b in enumerate(p)))

    def cmd(self, k, v=None):
        self.send({"t": "cmd", "k": k, "v": v})

    def inp(self, st=(0, 0), th=0.42, pd=0.0):
        self.send({"t": "in", "st": list(st), "pd": pd, "th": th, "brk": 0.0})


if __name__ == "__main__":
    c = WSC()
    what = sys.argv[1] if len(sys.argv) > 1 else "cockpit"
    if what == "cams":
        for cam in range(4):
            c.cmd("cam")
            jpg = c.grab(1.2)
            open("tools/view_%d.jpg" % cam, "wb").write(jpg)
            print("cam", cam, len(jpg))
    elif what == "night":
        c.cmd("tod", 0.95)
        jpg = c.grab(2.0)
        open("tools/view_night.jpg", "wb").write(jpg)
        print("night", len(jpg))
    elif what == "chase":
        c.cmd("cam")
        jpg = c.grab(1.5)
        open("tools/view_chase.jpg", "wb").write(jpg)
    else:
        jpg = c.grab(1.5)
        open("tools/view_cockpit.jpg", "wb").write(jpg)
        print("cockpit", len(jpg), c.last_state.get("res"), c.last_state.get("fps"))
