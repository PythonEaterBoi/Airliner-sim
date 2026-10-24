#!/usr/bin/env python3
"""AIRLINER — python-rendered 3D flight simulator.

Starts the streaming server.  Open the printed URL on your phone (same Wi-Fi
network) or on this machine; every pixel of the outside world is rasterised in
numpy on this box and sent as JPEG over a WebSocket.

    python3 run.py                 # http://0.0.0.0:8000
    python3 run.py --port 9000 --quality 1.25
"""
import argparse
import asyncio
import socket
import sys

from sim.server import main


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--quality", type=float, default=1.0,
                    help="terrain detail multiplier (0.6 low .. 1.25 high)")
    a = ap.parse_args()
    ip = lan_ip()
    print("\n  AIRLINER flight simulator (server-side python renderer)")
    print("  --------------------------------------------------------")
    print("  this machine :  http://localhost:%d" % a.port)
    print("  your phone   :  http://%s:%d   (same network)" % (ip, a.port))
    print("  landscape orientation recommended; touch controls on screen.\n",
          flush=True)
    try:
        asyncio.run(main(a.port, a.quality))
    except KeyboardInterrupt:
        sys.exit(0)
