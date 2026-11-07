"""Build a synthetic test clip from scikit-image's astronaut portrait.

This is NOT a substitute for a real recording; it only lets the whole
pipeline (decode -> MediaPipe -> stabiliser -> JSON) be exercised without a
camera. Phase 1: upright. Phase 2: image rolled by --roll degrees, which looks
like a sideways shoulder tilt. A few frames are blacked out to simulate
dropped/corrupt frames. --zoom != 1 crops/zooms to test scale invariance.

    pip install scikit-image   # dev-only, not in requirements.txt
    python tests/make_synthetic_video.py out.mp4 --seconds 30 --roll -14
"""
import argparse
import random

import cv2
import numpy as np
from skimage import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--roll", type=float, default=-14.0,
                    help="negative adds to the portrait's native tilt (positive cancels it)")
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--black-frames", type=int, default=6)
    a = ap.parse_args()
    base = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    h, w = base.shape[:2]
    rnd = random.Random(0)
    n = a.seconds * a.fps
    bad_idx = set(rnd.sample(range(n), a.black_frames))
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    for i in range(n):
        ang = 0.0 if i < n // 2 else a.roll
        ang += rnd.gauss(0, 0.4)                       # small hand-tremor jitter
        M = cv2.getRotationMatrix2D((w / 2, h * 0.6), ang, a.zoom)
        fr = cv2.warpAffine(base, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        if i in bad_idx:
            fr = np.zeros_like(fr)
        vw.write(fr)
    vw.release()


if __name__ == "__main__":
    main()
