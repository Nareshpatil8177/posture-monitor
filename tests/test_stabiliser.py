import random
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from posture_monitor import Config, Features, Baseline, naive_is_bad, Stabiliser, GOOD, BAD


BASE = Baseline(0.0, 0.0, 0.60)


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
        bad = naive_is_bad(f, BASE, c)
        if prev is not None and bad != prev:
            flips += 1
        prev = bad
    print("naive flips:", flips)
    # Ground truth has exactly 1 transition. Naive gives dozens.
    assert flips > 10


def run_stab(series, cfg=None, base=BASE):
    st = Stabiliser(cfg or Config(), base)
    for t, f, *_ in series:
        st.update(t, f)
    return st.samples


def transitions(samples):
    return [(round(a[0], 2), b[1]) for a, b in zip(samples, samples[1:]) if a[1] != b[1]]


def test_stabilised_single_transition():
    tr = transitions(run_stab(synthetic()))
    print("stabilised transitions:", tr)
    assert len(tr) == 1 and tr[0][1] == BAD
    assert abs(tr[0][0] - 20.0) < 1.0          # back-dated close to true onset


def test_short_blip_is_not_a_break():
    # 0.8 s head-drop in an otherwise good session must not register.
    s = []
    for i in range(30 * 20):
        t = i / 30
        neck = 0.60 * (0.8 if 8 <= t < 8.8 else 1.0)
        s.append((t, Features(2, 0.05, neck)))
    assert transitions(run_stab(s)) == []


def test_natural_asymmetry_is_not_slouching():
    # Someone who sits level-headed but with a steady 9 deg shoulder asymmetry.
    s = [(i / 30, Features(9.0, 0.10, 0.6)) for i in range(30 * 20)]
    with_base = run_stab(s, base=Baseline(9.0, 0.10, 0.6))
    assert all(x[1] == GOOD for x in with_base)
    # Without baseline subtraction the same person is "bad" the whole time.
    no_base = run_stab(s, base=Baseline(0.0, 0.0, 0.6))
    assert all(x[1] == BAD for x in no_base)


def test_gap_is_not_bad_posture():
    # 5 s with no detections: nothing may be labelled across the hole.
    s = [(i / 30, Features(2, 0.05, 0.6)) for i in range(30 * 5)]
    s += [(10 + i / 30, Features(2, 0.05, 0.6)) for i in range(30 * 5)]
    samples = run_stab(s)
    assert all(x[1] == GOOD for x in samples)


if __name__ == "__main__":
    for fn in (test_naive_flickers, test_stabilised_single_transition,
               test_short_blip_is_not_a_break, test_natural_asymmetry_is_not_slouching,
               test_gap_is_not_bad_posture):
        fn()
    print("all passed")
