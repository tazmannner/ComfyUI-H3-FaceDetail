"""Face tracks for a clip: per-frame detection -> shots (hard cuts) -> one smoothed track per character per shot.

A track holds, for every frame of its shot, the face box (smoothed) and a square crop box (centre smoothed, size
smoothed harder so the crop never 'breathes'). Frames where the face was not detected are interpolated and flagged."""
import numpy as np
import cv2


def shot_cuts(frames, threshold=0.45):
    """frames: (N,H,W,3) float. Returns list of shot start indices (always contains 0).
    A cut = colour histogram distance to the previous frame that is large and stands out from its neighbours."""
    n = len(frames)
    hs = []
    for f in frames:
        s = cv2.resize((np.clip(f, 0, 1) * 255).astype(np.uint8), (64, 36), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(s, cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        hs.append(h / (h.sum() + 1e-9))
    d = np.zeros(n)
    for i in range(1, n):
        d[i] = 0.5 * np.abs(hs[i] - hs[i - 1]).sum()
    starts = [0]
    for i in range(1, n):
        local = np.median(d[max(1, i - 6):min(n, i + 7)])
        if d[i] > threshold and d[i] > 3 * local + 0.1 and i - starts[-1] >= 6:
            starts.append(i)
    return starts


def _smooth(x, sigma):
    if sigma <= 0 or len(x) < 3:
        return x.copy()
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    xp = np.pad(x, (r, r), mode="edge")
    return np.convolve(xp, k, mode="valid")


def build_tracks(dets, starts, n, W, H, min_presence=0.1, max_faces=4, center_sigma=3.0, size_sigma=6.0, handle=6):
    """dets: per frame list of (x1,y1,x2,y2,score). Returns list of track dicts:
    {shot:(a,b), cx,cy,fw,fh: arrays over the shot (smoothed face box), found: bool array, side: smoothed crop side}"""
    tracks = []
    bounds = list(zip(starts, starts[1:] + [n]))
    for a, b in bounds:
        live = []  # [{frames:{i:box}}]
        for i in range(a, b):
            boxes = sorted(dets[i], key=lambda d: -d[4])
            used = set()
            for t in live:
                last_i = max(t["f"])
                if i - last_i > 12:
                    continue
                lb = t["f"][last_i]
                lc = np.array([(lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2]); ls = max(lb[3] - lb[1], lb[2] - lb[0])
                best, bd = None, 1e9
                for j, d in enumerate(boxes):
                    if j in used:
                        continue
                    c = np.array([(d[0] + d[2]) / 2, (d[1] + d[3]) / 2])
                    dist = np.linalg.norm(c - lc) / ls
                    if dist < 0.9 + 0.15 * (i - last_i) and dist < bd:
                        best, bd = j, dist
                if best is not None:
                    used.add(best)
                    t["f"][i] = boxes[best]
            for j, d in enumerate(boxes):
                if j not in used:
                    live.append({"f": {i: d}})
        L = b - a
        live = [t for t in live if len(t["f"]) >= max(8 if L > 8 else 1, min_presence * L)]
        live.sort(key=lambda t: -np.mean([d[3] - d[1] for d in t["f"].values()]) * len(t["f"]))
        for t in live[:max_faces]:
            idx = np.array(sorted(t["f"]))
            arr = np.array([t["f"][i][:4] for i in idx], dtype=np.float64)
            full = np.arange(a, b)
            cols = [np.interp(full, idx, arr[:, k]) for k in range(4)]  # holds the ends flat
            x1, y1, x2, y2 = cols
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            fs = np.maximum(x2 - x1, y2 - y1)
            found = np.isin(full, idx)
            found[1:-1] |= found[:-2] & found[2:]  # detection runs on every 2nd frame: bridge single-frame gaps
            # active only around where the face was actually seen (+handles for the fade); outside the slot is empty
            active = (full >= idx[0] - handle) & (full <= idx[-1] + handle)
            # centre smoothing only where the face is slow: on a fast head (a spin, a whip turn) the smoothed path cuts
            # the corner and the crop lands off the face, so there the (lightly smoothed) detections are followed
            fsm = _smooth(fs, size_sigma)
            speed = _smooth(np.hypot(np.gradient(cx), np.gradient(cy)), 1.0) / np.maximum(fsm, 1.0)  # face sizes / frame
            w = np.clip((speed - 0.05) / 0.10, 0, 1)
            scx = (1 - w) * _smooth(cx, center_sigma) + w * _smooth(cx, 0.7)
            scy = (1 - w) * _smooth(cy, center_sigma) + w * _smooth(cy, 0.7)
            tracks.append({"shot": (a, b), "cx": scx, "cy": scy, "fs": fsm, "found": found, "active": active,
                           "speed": speed})
    return tracks
