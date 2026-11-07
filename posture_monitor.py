#!/usr/bin/env python3
"""Seated posture monitor (work in progress: feature extraction only)."""
from __future__ import annotations

import math
from dataclasses import dataclass

# MediaPipe Pose landmark indices
L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 7, 8, 11, 12


@dataclass
class Features:
    """All three are dimensionless / angular, so they do not change when the
    person sits closer to the camera.

    tilt_deg   : shoulder-line roll from horizontal, degrees (0 = level)
    lateral    : |ear-midpoint x - shoulder-midpoint x| / shoulder width
                 (head sliding/leaning sideways relative to the torso)
    neck_ratio : distance(ear-mid, shoulder-mid) / shoulder width
                 (shrinks when the head drops / sinks into the shoulders)
    """
    tilt_deg: float
    lateral: float
    neck_ratio: float


def compute_features(lm, w: int, h: int) -> Features | None:
    """lm: sequence of landmarks with .x .y (normalised) .visibility.
    Convert to pixels first: normalised x and y have different scales on a
    non-square frame, which would distort angles and ratios."""
    def pt(i):
        return lm[i].x * w, lm[i].y * h

    lsx, lsy = pt(L_SHOULDER)
    rsx, rsy = pt(R_SHOULDER)
    lex, ley = pt(L_EAR)
    rex, rey = pt(R_EAR)
    width = math.hypot(lsx - rsx, lsy - rsy)
    if not math.isfinite(width) or width < 1e-6:
        return None
    # Angle of the shoulder line, folded so left/right labelling doesn't matter.
    ang = math.degrees(math.atan2(lsy - rsy, lsx - rsx))
    tilt = abs(((ang + 90) % 180) - 90)
    smx, smy = (lsx + rsx) / 2, (lsy + rsy) / 2
    emx, emy = (lex + rex) / 2, (ley + rey) / 2
    vals = Features(
        tilt_deg=tilt,
        lateral=abs(emx - smx) / width,
        neck_ratio=math.hypot(emx - smx, emy - smy) / width,
    )
    if not all(math.isfinite(v) for v in vals.__dict__.values()):
        return None
    return vals
