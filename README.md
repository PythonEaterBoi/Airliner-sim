# AIRLINER — a 3D flight simulator rendered entirely in Python

Every pixel of the outside world is rasterised **server-side by numpy** (a
software 3D pipeline: quaternion attitude → perspective projection → batched
scanline rasteriser with a perspective-correct 1/z depth buffer → exponential
fog → JPEG). Frames are streamed over a WebSocket to your **phone**, which
only displays the camera view and sends touch inputs back. No GPU, no WebGL,
no client-side 3D — the browser is a thin JPEG viewer with an HTML flight-deck
overlay.

```
        ┌──────────────────────────── server (python) ────────────────────────────┐
        │  6-DoF flight model → world/terrain → numpy rasteriser → JPEG ─→ WS ──┐ │
        │  autopilot / autothrottle / ILS        ↑                              │ │
        └────────────────────────────────────────┼──────────────────────────────┘ │
                                                 │ touch stick / sliders / buttons│
        ┌──────────────────────── phone browser ─┴───────────────────────────────┐│
        │  <canvas> JPEG frames + HTML HUD (tapes, ADI, LOC/GS, FMA) + controls  │◄┘
        └────────────────────────────────────────────────────────────────────────┘
```

## Run it

Requires Python 3.10+ with `numpy` and `Pillow` (or `opencv-python` for faster
JPEG encoding). Nothing else.

```bash
python3 run.py                 # serves http://0.0.0.0:8000
python3 run.py --port 9000 --quality 1.25
```

Open the printed LAN URL on your phone (same Wi-Fi), rotate to **landscape**,
tap once to unlock sound. On the same machine `http://localhost:8000` works
with keyboard + mouse.

## Flying

| Control | Touch | Keyboard |
|---|---|---|
| pitch / roll | drag left half of screen | arrows (up = nose up) |
| thrust | vertical slider, right edge (drag below 0 = reverse) | Shift / Ctrl |
| rudder / steering | orange slider | Q / E |
| gear, flaps, speedbrake | GEAR / FLAP± / SPDBRK | G, F / R, S |
| wheel & park brake | hold BRK / PARK | B / P |
| trim | TRIM↑↓ | T |
| autopilot / autothrottle / ILS arm | AP / A/T / APP | A / Z / X |
| camera (cockpit/chase/wing/tower) | CAM | V |
| menu: situation, time of day, quality | MENU | M |

**Takeoff:** thrust 100 %, rotate ≈145 kt, gear up at 400 ft, climb 160–180 kt.
**Landing:** arm APP on final (or hand-fly), gear down by 1 000 ft, flaps
15–30, 130–140 kt, flare ~30 ft, idle, brakes + speedbrake after touchdown.
The autopilot flies a full ILS: LOC capture, glideslope capture with an
auto-flare, autothrottle speed hold.

Situations: takeoff roll RWY 27, ILS final (10 NM, 2 000 ft), cruise FL240,
traffic pattern. Time of day slides from night (runway/city lights, emissive
windows) through dawn to noon.

## What is simulated

* **Flight model** — 6-DoF rigid body with quaternion attitude; lift/drag from
  α, flap and gear state; ground effect; stall & overspeed envelopes; engine
  spool, N1, thrust lapse with ρ and Mach; wheel brakes, parking brake,
  reverse thrust; speedbrakes; rate-limited control surfaces; auto trim
  integrator so the jet is flyable hands-off.
* **Warnings** — STALL, OVERSPEED, SINK RATE, PULL UP, TOO LOW GEAR/FLAPS,
  FLAP LIMIT, GEAR SPEED, with HUD master-warning and WebAudio alerts.
* **World** — 3 200 m runway 09/27 with markings, PAPI, approach lights,
  taxiways, apron, terminal, hangars, control tower, a 13×13-block city on a
  flattened plateau with a street grid, mountains, ocean, lake, value-noise
  terrain with LOD rings, volumetric-ish clouds, AI traffic with strobes,
  parked airliners. Runway/approach lights use distance-tiered LODs (flat
  quads close in, upright light bars far out) plus a depth-tested glow-dot
  splat pass, so the ILS stays visible from 10 km at night.
* **Renderer** — batched triangle rasteriser, perspective-correct depth,
  backface culling, per-tri flat shading + sun specular + fresnel on water,
  exp² distance fog, baked cockpit interior overlay, nav-light blips,
  adaptive internal resolution (320×180 … 640×360) to hold ~25–30 fps.

## Layout

```
run.py               launcher (prints LAN URL)
sim/mathx.py         quaternions, noise, rng, helpers
sim/terrain.py       value-noise heightfield + LOD ring meshes
sim/meshes.py        mesh builder: airliner, gear, cockpit, clouds, text
sim/world.py         airport, city, lights, traffic, scene graph
sim/renderer.py      numpy rasteriser, sky, fog, camera
sim/flightmodel.py   6-DoF aircraft + systems + warnings
sim/autopilot.py     AP / A/T / ILS LOC+GS capture + flare
sim/engine.py        sim driver, cameras, HUD state
sim/server.py        asyncio HTTP + RFC6455 WebSocket, JPEG streaming
static/index.html    mobile flight deck (HUD, touch sticks, menu)
tools/simtest.py     headless ILS autopilot regression test
tools/wsclient.py    WS harness (drives the sim like a phone)
tools/scene.py       offline scene screenshots
```

## Performance

On a 2-vCPU sandbox: ~30 ms/frame at 448×252 (≈600–1 500 triangles drawn),
25–30 fps streamed at ~5–10 KB JPEG per frame (~2 Mbps). The adaptive
resolution controller keeps the frame budget; `--quality` scales terrain LOD.
