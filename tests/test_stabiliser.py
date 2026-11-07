import random
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from posture_monitor import Config, Features, naive_is_bad


def synthetic(seconds=40, fps=30, seed=1):
    """Person sits well for 20 s then slouches for 20 s. Measurement noise
    is sized to be realistic for a pose model on a webcam: neck_ratio jitters
    by ~+-5% and tilt by ~+-2 deg, with the odd outlier frame."""
    rnd = random.Random(seed)
    out = []
    for i in range(seconds * fps):
        t = i / fps
        slouch = t >= 20
        neck = 0.60 * (0.85 if slouch else 1.0) + rnd.gauss(0, 0.025)
        tilt = (3.0 if not slouch else 4.0) + abs(rnd.gauss(0, 1.8))
        if rnd.random() < 0.01:                      # outlier frame
            neck *= 0.6
        out.append((t, Features(tilt, 0.05 + abs(rnd.gauss(0, 0.03)), neck), slouch))
    return out


def test_naive_flickers():
    c = Config()
    flips, prev = 0, None
    for _, f, _ in synthetic():
        bad = naive_is_bad(f, 0.60, c)
        if prev is not None and bad != prev:
            flips += 1
        prev = bad
    print("naive flips:", flips)
    # Ground truth has exactly 1 transition. Naive gives dozens.
    assert flips > 10


if __name__ == "__main__":
    test_naive_flickers()
