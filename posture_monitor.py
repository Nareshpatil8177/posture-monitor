#!/usr/bin/env python3
"""Seated posture monitor.

Reads a webcam or video file, tracks shoulders/ears with MediaPipe Pose,
stabilises a per-frame good/bad verdict over time, and writes a JSON summary.

Exit codes: 0 usable observation | 2 source cannot be opened |
            3 source opened but no frame could be decoded |
            4 nobody (or too little of anybody) in frame | 1 unexpected error
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field

# MediaPipe Pose landmark indices
L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 7, 8, 11, 12


@dataclass
class Features:
    """All three are dimensionless / angular, so they do not change when the
    person sits closer to the camera.

    tilt_deg   : SIGNED shoulder-line roll from horizontal, degrees
    lateral    : SIGNED (ear-midpoint x - shoulder-midpoint x) / shoulder width
                 (head sliding/leaning sideways relative to the torso)
    Tilt and lateral are signed so a personal baseline can be subtracted:
    most people are a few degrees asymmetric even when sitting well.
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
    tilt = ((ang + 90) % 180) - 90      # (-90, 90]; folding makes L/R labelling irrelevant
    smx, smy = (lsx + rsx) / 2, (lsy + rsy) / 2
    emx, emy = (lex + rex) / 2, (ley + rey) / 2
    vals = Features(
        tilt_deg=tilt,
        lateral=(emx - smx) / width,
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
    # enter-bad / exit-bad (hysteresis: exit is stricter than enter).
    # tilt / lateral are |deviation from your own baseline|; drop is the
    # fractional shrink of neck_ratio vs baseline.
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


@dataclass
class Baseline:
    """Your own upright reference, from the calibration window."""
    tilt_deg: float = 0.0
    lateral: float = 0.0
    neck_ratio: float = 0.6


def deviations(f: Features, b: Baseline) -> tuple[float, float, float]:
    """-> (|tilt dev| deg, |lateral dev|, neck drop fraction; + = head sinking)"""
    return (abs(f.tilt_deg - b.tilt_deg), abs(f.lateral - b.lateral),
            1.0 - f.neck_ratio / b.neck_ratio)


def naive_is_bad(f: Features, b: Baseline, c: Config) -> bool:
    """Per-frame verdict with a single hard line per metric. Kept for the
    tests (to quantify flicker); the real path goes through Stabiliser."""
    t, l, d = deviations(f, b)
    return t > c.tilt_enter or l > c.lateral_enter or d > c.drop_enter


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

    def __init__(self, cfg: Config, baseline: Baseline):
        self.c, self.base = cfg, baseline
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
        tilt, lat, drop = deviations(Features(tilt, lat, neck), self.base)
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


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
def _pct(vals, q):
    if not vals:
        return None
    v = sorted(vals)
    k = (len(v) - 1) * q / 100
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 4)


def summarise(samples, observed_s, cfg: Config) -> dict:
    """samples: [[t, state], ...] over *valid* frames only."""
    good = bad = 0.0
    runs = []                      # [state, start, end]
    for i, (t, st) in enumerate(samples):
        nxt = samples[i + 1][0] if i + 1 < len(samples) else t
        dt = nxt - t
        contiguous = dt <= cfg.max_gap_s
        span = dt if contiguous else 0.0
        if st == GOOD:
            good += span
        else:
            bad += span
        if runs and runs[-1][0] == st and runs[-1][2] >= t - 1e-9:
            runs[-1][2] = t + span
        else:
            runs.append([st, t, t + span])
    valid = good + bad
    breaks = sum(1 for a, b in zip(runs, runs[1:]) if a[0] == GOOD and b[0] == BAD)
    bad_runs = [r[2] - r[1] for r in runs if r[0] == BAD]
    good_runs = [r[2] - r[1] for r in runs if r[0] == GOOD]
    good_frac = good / valid if valid > 0 else 0.0
    per_min = breaks / (valid / 60) if valid > 0 else 0.0
    score = max(0.0, min(100.0, 100 * good_frac - min(20.0, 4.0 * per_min)))
    r = lambda x: round(x, 2)
    return {
        "good_seconds": r(good), "bad_seconds": r(bad),
        "unknown_seconds": r(max(0.0, observed_s - valid)),
        "good_pct": r(100 * good_frac), "bad_pct": r(100 * (1 - good_frac)) if valid else 0.0,
        "breaks": {
            "count": breaks, "per_minute": r(per_min),
            "mean_bad_episode_s": r(statistics.mean(bad_runs)) if bad_runs else 0.0,
            "longest_bad_s": r(max(bad_runs)) if bad_runs else 0.0,
            "longest_good_s": r(max(good_runs)) if good_runs else 0.0,
        },
        "score": round(score),
        "episodes": [{"state": s_, "start_s": r(a), "end_s": r(b)} for s_, a, b in runs],
    }


# --------------------------------------------------------------------------
# Capture + CLI
# --------------------------------------------------------------------------
EXIT_OK, EXIT_ERROR, EXIT_NO_SOURCE, EXIT_NO_FRAMES, EXIT_NO_PERSON = 0, 1, 2, 3, 4
MAX_CONSEC_READ_FAILS = 30
MIN_SHOULDER_VIS, MIN_EAR_VIS = 0.5, 0.3


@dataclass
class FrameStats:
    attempted: int = 0
    read_failed: int = 0     # cap.read() returned nothing
    corrupt: int = 0         # decoded but unusable (None/empty/wrong shape)
    errors: int = 0          # exception inside the pose model / feature code
    no_person: int = 0
    low_visibility: int = 0
    valid: int = 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Seated posture monitor")
    ap.add_argument("--source", default="0", help="camera index or video path (default 0)")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds of video to analyse")
    ap.add_argument("--out", default="posture.json")
    ap.add_argument("--preview", action="store_true", help="optional window; q quits")
    ap.add_argument("--calib-seconds", type=float, default=8.0,
                    help="first N seconds of valid data define your personal upright baseline")
    ap.add_argument("--baseline-neck", type=float, default=None,
                    help="skip calibration and use this neck_ratio as upright baseline")
    ap.add_argument("--min-coverage", type=float, default=0.3,
                    help="fraction of the observed time that must contain a usable person")
    ap.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=1)
    ap.add_argument("--dump-features", metavar="CSV", help="write per-frame features (for threshold review)")
    return ap.parse_args(argv)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)             # atomic: a retrying pipeline never sees half a file


def fail(args, code, reason, stats=None, extra=None):
    print(f"posture_monitor: FAILED ({reason})", file=sys.stderr)
    body = {"schema_version": 1, "status": "failed", "reason": reason, "exit_code": code,
            "source": args.source, "frames": asdict(stats) if stats else None}
    body.update(extra or {})
    try:
        write_json(args.out, body)
    except OSError as e:
        print(f"posture_monitor: could not write {args.out}: {e}", file=sys.stderr)
    return code


def capture(args, stop):
    """Yield (timestamp_s, frame_or_None). None marks a failed/corrupt read.
    File sources use frame_index/fps as the clock (so analysis can run faster
    than real time); live cameras use the monotonic wall clock."""
    import cv2
    is_cam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_cam else args.source)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 and math.isfinite(fps) else 30.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    t0 = time.monotonic()

    def gen():
        idx = fails = 0
        try:
            while not stop["flag"]:
                ts = (time.monotonic() - t0) if is_cam else idx / fps
                if ts >= args.duration:
                    return
                try:
                    ok, frame = cap.read()
                except Exception:
                    ok, frame = False, None
                idx += 1
                if not ok:
                    fails += 1
                    if not is_cam and n_frames and idx >= n_frames:
                        return                     # clean end of file
                    if fails >= MAX_CONSEC_READ_FAILS:
                        return
                    if is_cam:
                        time.sleep(0.03)
                    yield ts, None
                    continue
                fails = 0
                yield ts, frame
        finally:
            cap.release()
    return gen()


def analyse(args, stop):
    """Pass 1: frames -> per-frame features. Returns (series, stats, observed_s) or None."""
    import cv2
    import mediapipe as mp
    stats = FrameStats()
    series = []                               # (t, Features)
    frames = capture(args, stop)
    if frames is None:
        return None
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=args.model_complexity,
                                  smooth_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    last_t, preview_ok = 0.0, args.preview
    try:
        for ts, frame in frames:
            last_t = ts
            stats.attempted += 1
            if frame is None:
                stats.read_failed += 1
                continue
            if getattr(frame, "ndim", 0) != 3 or frame.shape[0] < 16 or frame.shape[1] < 16:
                stats.corrupt += 1
                continue
            try:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                res = pose.process(rgb)
                feat = None
                if res.pose_landmarks is None:
                    stats.no_person += 1
                else:
                    lm = res.pose_landmarks.landmark
                    if (min(lm[L_SHOULDER].visibility, lm[R_SHOULDER].visibility) < MIN_SHOULDER_VIS
                            or min(lm[L_EAR].visibility, lm[R_EAR].visibility) < MIN_EAR_VIS):
                        stats.low_visibility += 1
                    else:
                        feat = compute_features(lm, w, h)
                        if feat is None:
                            stats.corrupt += 1
                        else:
                            stats.valid += 1
                            series.append((ts, feat))
            except Exception as e:             # never let one bad frame kill the run
                stats.errors += 1
                feat = None
                if stats.errors <= 3:
                    print(f"posture_monitor: frame error ignored: {e!r}", file=sys.stderr)
            if preview_ok:
                try:
                    txt = (f"tilt {feat.tilt_deg:.1f} lat {feat.lateral:.2f} neck {feat.neck_ratio:.2f}"
                           if feat else "no usable pose")
                    cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("posture", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop["flag"] = True
                except cv2.error:
                    preview_ok = False
                    print("posture_monitor: no display available, preview disabled", file=sys.stderr)
    finally:
        pose.close()
        if args.preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
    return series, stats, last_t


def calibrate(series, args):
    """Personal upright baseline = medians over the first calib window of valid
    data. Returns (Baseline, info)."""
    t0 = series[0][0]
    win = [f for t, f in series if t - t0 <= args.calib_seconds]
    med = lambda k: statistics.median(getattr(f, k) for f in win)
    base = Baseline(med("tilt_deg"), med("lateral"), med("neck_ratio"))
    src = "first_valid_seconds"
    if args.baseline_neck:
        base.neck_ratio, src = args.baseline_neck, "neck from cli, tilt/lateral from first_valid_seconds"
    nr = [f.neck_ratio for f in win]
    spread = (_pct(nr, 90) - _pct(nr, 10)) / base.neck_ratio
    tilt_sd = statistics.pstdev([f.tilt_deg for f in win]) if len(win) > 1 else 0.0
    return base, {"source": src, "window_s": args.calib_seconds, "frames": len(win),
                  "baseline_tilt_deg": round(base.tilt_deg, 2),
                  "baseline_lateral": round(base.lateral, 4),
                  "baseline_neck_ratio": round(base.neck_ratio, 4),
                  "neck_spread_p10_p90": round(spread, 3), "tilt_sd_deg": round(tilt_sd, 2)}


def run(args) -> int:
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    out = analyse(args, stop)
    if out is None:
        return fail(args, EXIT_NO_SOURCE, "cannot open source")
    series, stats, observed = out
    if stats.attempted == 0 or stats.attempted == stats.read_failed:
        return fail(args, EXIT_NO_FRAMES, "source opened but no frame could be decoded", stats)
    cfg = Config()
    valid_span = (series[-1][0] - series[0][0]) if len(series) > 1 else 0.0
    # coverage = share of the observed timeline that had a usable person
    coverage = (stats.valid / stats.attempted) if stats.attempted else 0.0
    if len(series) < 5 or coverage < args.min_coverage or valid_span < 2.0:
        return fail(args, EXIT_NO_PERSON,
                    f"no usable person in frame (coverage {coverage:.0%}, need {args.min_coverage:.0%})",
                    stats, {"coverage": round(coverage, 3)})

    base, calib = calibrate(series, args)
    st = Stabiliser(cfg, base)
    for t, f in series:
        st.update(t, f)
    summ = summarise(st.samples, observed, cfg)

    warnings = []
    if calib["neck_spread_p10_p90"] > 0.15 or calib["tilt_sd_deg"] > 3.0:
        warnings.append("calibration window was unsteady; baseline may be unreliable "
                        "(person moving, or slouching during calibration)")
    if stats.errors:
        warnings.append(f"{stats.errors} frames raised errors and were skipped")
    if stop["flag"]:
        warnings.append("interrupted early; summary covers the observed part only")

    feats = {k: {"p5": _pct([getattr(f, k) for _, f in series], 5),
                 "p50": _pct([getattr(f, k) for _, f in series], 50),
                 "p95": _pct([getattr(f, k) for _, f in series], 95)}
             for k in ("tilt_deg", "lateral", "neck_ratio")}
    body = {
        "schema_version": 1, "status": "ok", "source": args.source,
        "observed_seconds": round(observed, 2), "coverage": round(coverage, 3),
        "score": summ.pop("score"),
        "posture": {k: summ[k] for k in ("good_seconds", "bad_seconds", "unknown_seconds", "good_pct", "bad_pct")},
        "breaks": summ["breaks"], "episodes": summ["episodes"],
        "frames": asdict(stats), "calibration": calib,
        "thresholds": asdict(cfg), "feature_percentiles": feats, "warnings": warnings,
    }
    write_json(args.out, body)
    if args.dump_features:
        with open(args.dump_features, "w") as fh:
            fh.write("t,tilt_deg,lateral,neck_ratio\n")
            for t, f in series:
                fh.write(f"{t:.3f},{f.tilt_deg:.3f},{f.lateral:.4f},{f.neck_ratio:.4f}\n")
    print(f"posture_monitor: score={body['score']} good={body['posture']['good_pct']}% "
          f"breaks={body['breaks']['count']} -> {args.out}")
    return EXIT_OK


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as e:
        return fail(args, EXIT_ERROR, f"unexpected error: {e!r}")


if __name__ == "__main__":
    sys.exit(main())
