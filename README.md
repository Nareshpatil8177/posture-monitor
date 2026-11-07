# Seated posture monitor

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python posture_monitor.py --source 0 --duration 60 --out posture.json            # webcam
python posture_monitor.py --source session.mp4 --out posture.json                # file (runs faster than real time)
python posture_monitor.py --source 0 --duration 45 --preview --dump-features features.csv   # optional window + CSV
python tests/test_stabiliser.py                                                  # unit tests, no camera needed
```
Headless and CPU-only by default. Sit upright for the first ~8 s: that is your calibration. Output: `status`, `score` (0–100),
`posture` (good/bad/unknown seconds and %), `breaks` (count, per minute, mean/longest bad episode), `episodes` timeline,
frame counters, calibration, thresholds used, feature percentiles, warnings.
**Exit codes:** 0 usable · 2 source can't be opened · 3 opened but nothing decodable · 4 nobody (or <30 % of the time) in frame · 1 unexpected.
Failures also write a `status:"failed"` JSON (atomically), so a retrying pipeline never reads a half file.

## Decisions
**Features** (shoulders + ears only; hips are rarely visible when seated): shoulder-line roll, ear-vs-shoulder sideways offset,
and ear-to-shoulder distance — the last two divided by shoulder width, so moving closer changes nothing. Computed in pixels,
not normalised coords (those distort angles on non-square frames). I deliberately did *not* use shoulder width itself: it grows when you lean in, which is exactly what scale-invariance forbids.

**Personal baseline.** My first synthetic run showed that even "good" posture had ~5° shoulder asymmetry, so absolute
thresholds aren't per-person. All three metrics are now deviations from the median of the first 8 s of valid data.
Cost: slouching during calibration poisons the baseline (a warning fires if the window is unsteady).

**Stabilisation** (the part I spent most time on; `Stabiliser`), four layers, each answering one failure:
1. EMA on the features, time-constant 0.6 s (in seconds, not frames, so fps-independent) — jitter and outlier frames.
2. Hysteresis — enter and exit thresholds differ — stops flapping on a line.
3. Minimum dwell, back-dated — 2 s to turn bad, 1 s to recover; a flip is only committed after it persisted, then dated from its start.
   Asymmetric because glancing around or nodding shouldn't count, but fixing your posture should be credited quickly.
4. Gaps — a hole >1 s resets the filters and counts as `unknown`, never as bad. Lost tracking isn't slouching.
On a synthetic noisy session (one true transition) the naive per-frame classifier flips **240×**; the stabilised one **1×**, ~0.9 s late.

**Thresholds — honest account.** I had no labelled human data and no webcam recording inside the time box, so these are *priors*, not fitted values:
enter-bad at tilt 8°, lateral 0.30 shoulder-widths, neck-ratio drop 12 %; exit at 5° / 0.20 / 7 % (~60 % of enter).
Evidence I do have (synthetic clip, so only the *noise floor* is informative): tilt σ≈1.4°, neck-ratio p10–p90 spread ≈3 %.
8° is ~5σ and 12 % is ~4× the spread, so noise alone should rarely cross. Where the line sits above that is judgement
(roughly "visible to a human"). **The neck-drop line is the one most likely wrong:** a frontal camera sees slouch mostly as the head sinking, and I haven't validated 12 % on a real slouch.
Use `--dump-features` plus `feature_percentiles` in the JSON (sitting-well p5/p95 vs slouch) to re-set them; they live in `Config`.
The synthetic test also showed threshold sensitivity directly: a 14° roll deviated only 6–8° from baseline and was detected 5 s late.

**Score** = 100 × good share of *valid* time − min(20, 4 × breaks/min). Good share dominates; repeated breaks cost up to 20 points.

## Not handled / known weaknesses
- **No real recording is included.** `examples/synthetic_run.json` is built from a still portrait (see `tests/make_synthetic_video.py`) and is labelled as such. Record yourself and commit a real `posture.json`.
- Calibration assumes you start upright; no re-calibration. Camera moving mid-session looks like a posture change.
- Frontal view only; forward lean without head drop is nearly invisible. Tilt is unreliable when shoulders touch the frame edge (saw 12.9° upright at 1.3× zoom; baseline absorbs a constant offset, not a changing one).
- Multiple people: MediaPipe picks one, no identity tracking. Mirrored cameras are fine (signs flip, deviations don't).
- Preview shows live raw features only — the verdict needs the baseline, which is computed after capture (two-pass).
- Pinned `mediapipe==0.10.21`: newer releases removed `mp.solutions`. Tested on Python 3.12, Linux, no GPU/camera hardware.
