#!/usr/bin/env bash
# Failure-mode / exit-code checks. Needs: pip install scikit-image (dev only).
# Expected: empty->4, garbage->2, missing camera->2, corrupt-midstream->0
set -u
cd "$(dirname "$0")/.."
T=$(mktemp -d)
python tests/make_synthetic_video.py $T/ok.mp4 --seconds 20
python - "$T" <<'PY'
import sys, numpy as np, cv2, random
T = sys.argv[1]
vw = cv2.VideoWriter(f"{T}/empty.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (320, 240))
for _ in range(150): vw.write(np.full((240, 320, 3), 90, np.uint8))
vw.release()
cap = cv2.VideoCapture(f"{T}/ok.mp4"); vw = cv2.VideoWriter(f"{T}/mj.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15, (512, 512))
while True:
    ok, f = cap.read()
    if not ok: break
    vw.write(f)
vw.release()
b = bytearray(open(f"{T}/mj.avi", "rb").read()); r = random.Random(3)
for _ in range(4):
    s = r.randrange(len(b) // 3, 2 * len(b) // 3); b[s:s + 3000] = bytes(r.randrange(256) for _ in range(3000))
open(f"{T}/mj_corrupt.avi", "wb").write(b)
PY
echo "not a video" > $T/garbage.mp4
check() { python posture_monitor.py --source "$1" --duration 20 --out $T/o.json >/dev/null 2>&1; echo "$2: exit $? (want $3)"; }
check $T/ok.mp4 "valid clip" 0
check $T/empty.mp4 "empty room" 4
check $T/garbage.mp4 "garbage file" 2
check 7 "no such camera" 2
check $T/mj_corrupt.avi "corrupt mid-stream" 0
