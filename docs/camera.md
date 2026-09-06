# The camera

**The frame rate is a format choice, not a camera limit.** A UVC webcam is opened for MJPG at
1280×720, 15 fps. Ask for no format and V4L2 hands over its first one — uncompressed YUYV — which
at 720p is 1.8 MB a frame. A USB 2.0 bus carries about 60 MB/s, so the camera negotiates itself
down to exactly 10 fps and no code ever mentions a frame rate. That was the old behaviour, and it
is why the preview stepped: the panel redraws about 25 times a second and only had ten new frames
to draw. MJPG frames are a tenth the size, the bus stops deciding, and `CAP_PROP_FPS` is then
honoured exactly. Measured on a Pi 5 at 720p:

| format | fps | read + decode | focus scoring | one reader thread |
|---|---|---|---|---|
| YUYV   | 10 (bus-capped) | 5.7 ms | ~4 ms | ~10% of a core |
| MJPG   | 10 | 8.4 ms | 4.3 ms | 13% of a core |
| MJPG   | 15 | 8.1 ms | 4.4 ms | 19% of a core |
| MJPG   | 30 | 5.7 ms | 3.6 ms | 28% of a core |
| MJPG 1080p | 30 | 14.4 ms | — | 43% of a core |

Whole-kiosk cost of the switch, same measurement both ways: 83% → 93% of one core, panel
temperature unchanged. The frames buy two things — a preview that fills most of the panel's
redraws, and half again as many candidates for `CameraSource.snapshot()`, which hands out the
sharpest frame of the last 0.7 s rather than the newest.

That last point ties `FRAME_RATE` in `webcam.py` to `HISTORY` in `camera.py`: the history is a
frame *count* and the window it feeds is a *duration*. Twelve frames is 0.8 s at 15 fps, which
just covers the 0.7 s window. Raise the rate to 30 without raising `HISTORY` and the picker only
ever sees the last 0.4 s — more frames, but a smaller slice of time to pick from.

## Cameras that aren't webcams

Most USB cameras are UVC devices: the kernel binds them, a `/dev/video*` node appears, and
`CYCLOPS_CAMERA_INDEX=auto` finds them. Some aren't. The cheap endoscopes sold as *supercamera*
(Geek szitman, and the Oasis/Depstech rebadges) expose two vendor-specific USB interfaces and
speak a proprietary protocol to a phone app, so no kernel driver binds them and **no
`/dev/video*` node is ever created** — they sit on the bus repeating a heartbeat at a host that
never answers. `lsusb` shows them, `v4l2-ctl --list-devices` does not, and every camera probe
comes up empty.

cyclops drives them anyway, from userspace over libusb. Once the `/dev/video*` probe finds
nothing, `open_camera()` looks for a useeplus endoscope and wraps it in the same
`read()`/`release()` shape the rest of the code already expects, so nothing above it knows the
difference — preview, SNAP and a `camera` recording all behave as usual, at 640×480. A real
webcam still wins if one is plugged in. The startup line says which you got:

```
· cyclops kiosk on camera useeplus     # the endoscope, over libusb
· cyclops kiosk on camera 0            # an ordinary UVC webcam at /dev/video0
```

The raw USB device is root-only by default, so access has to be granted: `deploy/push.sh`
installs `deploy/99-useeplus-camera.rules`, handing the device to group `video` — the one the
kiosk user is already in for `/dev/video*`. Without that rule the kiosk finds no camera at all.
A few `Corrupt JPEG data` lines at startup are the connect handshake and are expected; a steady
stream of them is not.

There is also an out-of-tree V4L2 kernel module for these cameras, which would produce a real
`/dev/video0` and need no cyclops code whatsoever. It is deliberately not used: it ships without
DKMS, so it stops loading the next time the kernel updates. On a box whose whole point is being
left alone, a camera that dies on `apt upgrade` is worse than sixty lines of Python.

**640×480 is the ceiling, whatever the box says.** These are sold as "1920P HD" — the listing for
this one claims 1920×1440, which is exactly 640×480 × 3 in each axis: it describes what the phone
app upscales to, not what the sensor reads out. The device streams 640×480 JPEG at quality ≈44
(~19 KB a frame, 0.45 bits per pixel) at 20 fps, and there is no way to ask it for more:

* The protocol's whole vocabulary is four commands — open (`BB AA 05 00 00`), ack, stop
  (`BB AA 08 00 00`), and switch camera/resolution (`BB AA 0B 00 02 <cam> <res> 00`). There is no
  still-capture command, so there is no high-resolution photo path hiding behind the video one.
* This unit ignores the resolution command. Every `cam`×`res` combination was tried against it,
  both mid-stream and before the stream is opened; it never acknowledges `0B` and never emits
  anything but 640×480 in type-`0x07` packets.
* Nothing is negotiable in the descriptors either: two vendor-specific interfaces, four bulk
  endpoints, one alternate setting, and no format descriptors at all.

Nor is the softness a focus problem, which is the intuitive diagnosis and the wrong one. A hard
edge takes the same 5 px to transition at 0.5 m as it does at 2.5 m — a fixed-focus lens that is
equally soft everywhere, not a focal plane you are standing outside. Distant things look worse
purely because the same pixels are spread over the same angle: at five times the distance a
feature is five times smaller, and it falls below what 640×480 and a quality-44 encoder can carry.

## Sharpening

The only lever left is downstream. `overlay.sharpen()` is a threshold- and ceiling-gated unsharp
mask applied to the preview before it is enlarged onto the panel, and to every photo before it is
encoded for the model. **It is switched off at the moment** — `overlay.SHARPEN` is `False`, which
makes the function a no-op on both paths; the rest of this section describes what turning it back
on does.

On a real frame it takes the Laplacian variance from 18.5 to 50.1 — 2.7× the detail — while the
noise floor in flat areas moves 2.10 to 2.20. The gates are what make that trade good: the floor
leaves detail weaker than the encoder's own blocking alone, and the ceiling caps how far any
pixel may travel, which is what stops a face against a bright window growing a white halo. It
measured 5.98 ms a frame against 5.42 ms before, on a 25 fps loop with 40 ms to spend.

A `camera` recording is untouched by it — the recorder samples raw frames straight from the
device — but a `screen` one is not, and should not be: it is the glass, and the glass is
sharpened.
