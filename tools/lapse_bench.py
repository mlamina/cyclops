"""Per-frame cost of the two ways to show the panel a 2304x1296 YUV420 frame (arc-eng/hq#669)."""

import time

import cv2
import numpy as np

W, H, SW, SH = 2304, 1296, 1280, 720
plane = np.random.randint(0, 255, (H * 3 // 2, W), dtype=np.uint8)


def convert_then_shrink():
    return cv2.resize(cv2.cvtColor(plane, cv2.COLOR_YUV2BGR_I420), (SW, SH), interpolation=cv2.INTER_AREA)


def shrink_then_convert():
    y = cv2.resize(plane[:H], (SW, SH), interpolation=cv2.INTER_AREA)
    u = cv2.resize(plane[H : H + H // 4].reshape(H // 2, W // 2), (SW // 2, SH // 2), interpolation=cv2.INTER_AREA)
    v = cv2.resize(plane[H + H // 4 :].reshape(H // 2, W // 2), (SW // 2, SH // 2), interpolation=cv2.INTER_AREA)
    small = np.vstack([y, u.reshape(-1, SW), v.reshape(-1, SW)])
    return cv2.cvtColor(small, cv2.COLOR_YUV2BGR_I420)


def native_720():
    return cv2.cvtColor(plane[: SH * 3 // 2, :SW].copy(), cv2.COLOR_YUV2BGR_I420)


def shrink_linear_then_convert():
    f = cv2.INTER_LINEAR
    y = cv2.resize(plane[:H], (SW, SH), interpolation=f)
    u = cv2.resize(plane[H : H + H // 4].reshape(H // 2, W // 2), (SW // 2, SH // 2), interpolation=f)
    v = cv2.resize(plane[H + H // 4 :].reshape(H // 2, W // 2), (SW // 2, SH // 2), interpolation=f)
    return cv2.cvtColor(np.vstack([y, u.reshape(-1, SW), v.reshape(-1, SW)]), cv2.COLOR_YUV2BGR_I420)


def join_chunks():
    raw = plane.tobytes()
    chunks = [raw[i : i + 65536] for i in range(0, len(raw), 65536)]
    return np.frombuffer(b"".join(chunks), np.uint8)


for fn in (native_720, convert_then_shrink, shrink_then_convert, shrink_linear_then_convert, join_chunks):
    fn()
    t = time.perf_counter()
    for _ in range(60):
        fn()
    print(f"{fn.__name__}: {(time.perf_counter() - t) / 60 * 1000:.1f} ms a frame")
