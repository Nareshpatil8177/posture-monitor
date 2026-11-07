#!/usr/bin/env bash
# Same movement rendered at three zoom levels; ratios should not move.
cd "$(dirname "$0")/.."
for z in 0.8 1.0 1.3; do
  python tests/make_synthetic_video.py /tmp/z$z.mp4 --zoom $z --roll -25
  python posture_monitor.py --source /tmp/z$z.mp4 --duration 30 --out /tmp/z$z.json
done
