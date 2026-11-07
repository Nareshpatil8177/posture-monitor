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


# --------------------------------------------------------------------------
# Thresholds (see README for how these were chosen)
# --------------------------------------------------------------------------
@dataclass
class Config:
    # enter-bad / exit-bad (hysteresis: exit is stricter than enter)
    tilt_enter: float = 8.0      # degrees
    tilt_exit: float = 5.0
    lateral_enter: float = 0.30  # fraction of shoulder width
    lateral_exit: float = 0.20
    drop_enter: float = 0.12     # neck_ratio fell >12% below personal baseline
    drop_exit: float = 0.07
    # temporal stabilisation
    ema_tau_s: float = 0.6       # smoothing time constant
    dwell_bad_s: float = 2.0     # must look bad this long before we say bad
    dwell_good_s: float = 1.0    # must look good this long before we say good again
    max_gap_s: float = 1.0       # a hole in the data longer than this is "unknown"


def naive_is_bad(f: Features, baseline_neck: float, c: Config) -> bool:
    """Per-frame verdict with a single hard line per metric. Kept for the
    tests (to quantify flicker) and as the input to the stabiliser."""
    drop = 1.0 - f.neck_ratio / baseline_neck
    return f.tilt_deg > c.tilt_enter or f.lateral > c.lateral_enter or drop > c.drop_enter


# --------------------------------------------------------------------------
# Stabilisation
# --------------------------------------------------------------------------
GOOD, BAD = "good", "bad"


class Stabiliser:
    """Turns noisy per-frame features into a committed GOOD/BAD timeline.

    Four layers, each answering a different failure:
      1. time-aware EMA on the *features*  -> jitter / outlier frames
      2. hysteresis (enter != exit)        -> hovering right on a threshold
      3. minimum dwell, back-dated         -> short real-looking blips
         (a flip is only committed after the new state has persisted for
         dwell seconds, and is then dated from when it *started*)
      4. gap handling                      -> lost tracking is not bad posture
         (a gap > max_gap_s resets the filters and is reported as unknown)
    Smoothing is by seconds, not frames, so webcam fps does not matter.
    """

    def __init__(self, cfg: Config, baseline_neck: float):
        self.c, self.base = cfg, baseline_neck
        self.samples: list[list] = []     # [t, committed_state]
        self._reset()

    def _reset(self):
        self.ema = None
        self.last_t = None
        self.raw_bad = False
        self.committed = None
        self.cand = None
        self.cand_since = None

    def _smooth(self, t, f: Features) -> tuple[float, float, float]:
        cur = (f.tilt_deg, f.lateral, f.neck_ratio)
        if self.ema is None:
            self.ema = list(cur)
        else:
            a = 1 - math.exp(-(t - self.last_t) / self.c.ema_tau_s)
            self.ema = [e + a * (x - e) for e, x in zip(self.ema, cur)]
        return tuple(self.ema)

    def _hysteresis(self, tilt, lat, neck) -> bool:
        c = self.c
        drop = 1.0 - neck / self.base
        if not self.raw_bad:
            if tilt > c.tilt_enter or lat > c.lateral_enter or drop > c.drop_enter:
                self.raw_bad = True
        else:
            if tilt < c.tilt_exit and lat < c.lateral_exit and drop < c.drop_exit:
                self.raw_bad = False
        return self.raw_bad

    def update(self, t: float, f: Features) -> str:
        if self.last_t is not None and t - self.last_t > self.c.max_gap_s:
            self._reset()                       # layer 4: don't bridge a hole
        raw = BAD if self._hysteresis(*self._smooth(t, f)) else GOOD
        self.last_t = t
        self.samples.append([t, raw if self.committed is None else self.committed])
        if self.committed is None:
            self.committed = raw
        elif raw == self.committed:
            self.cand = None                    # blip ended: forget candidate
        else:
            if self.cand != raw:
                self.cand, self.cand_since = raw, t
            dwell = self.c.dwell_bad_s if raw == BAD else self.c.dwell_good_s
            if t - self.cand_since >= dwell:
                self.committed = raw
                for s in reversed(self.samples):   # back-date to when it began
                    if s[0] < self.cand_since:
                        break
                    s[1] = raw
                self.cand = None
        return self.committed
