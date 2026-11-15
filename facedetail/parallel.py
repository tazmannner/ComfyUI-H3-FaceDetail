"""Face Atlas Stitch on CPU threads: the same arithmetic as atlas.py's stitch, geometry_lock and fidelity, with the
per-tile / per-frame loops spread over a thread pool. OpenCV and numpy release the GIL, every task writes its own pixels
and results are gathered in the original order, so the output is bit-identical to the single-threaded functions (which
stay in atlas.py as the reference). About 3x faster on 8 cores."""
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

WORKERS = max(1, min(8, (os.cpu_count() or 4) // 2))


def pmap(fn, items, workers=None):
    """fn over items on a thread pool, results in order. OpenCV's own thread pool is switched off meanwhile: the pool
    already uses the cores, and both together run several times slower (16 x 8 threads fighting over 8 cores)."""
    items = list(items)
    w = WORKERS if workers is None else workers
    if w <= 1 or len(items) < 2:
        return [fn(x) for x in items]
    prev = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ThreadPoolExecutor(w) as ex:
            return list(ex.map(fn, items))
    finally:
        cv2.setNumThreads(prev)


def make(atlas_mod, track_mod):
    """-> (stitch, geometry_lock, fidelity) bound to the pack's helper functions"""
    _crop_matrix, _tile_mask, _smooth = atlas_mod._crop_matrix, atlas_mod._tile_mask, track_mod._smooth

    def _tiles(lay):
        return [(k, i) for k in range(lay["nslots"]) for i in range(lay["n"]) if lay["slots"][k][i] is not None]

    def fidelity(frames, refined, lay, res=40, full=1.8, zero=2.5, workers=None):
        T = lay["T"]
        yy, xx = np.mgrid[:res, :res]
        ell = ((xx - res / 2) / (res * 0.29)) ** 2 + ((yy - res / 2) / (res * 0.32)) ** 2 < 1

        def one(ki):
            k, i = ki
            ox, oy = lay["origin"][k]
            t, j = lay["slots"][k][i]
            M = _crop_matrix(t["cx"][j], t["cy"][j], t["side"][j], T)
            orig = cv2.warpAffine(frames[i], M, (T, T), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
            a = cv2.GaussianBlur(cv2.resize(orig, (res, res), interpolation=cv2.INTER_AREA), (0, 0), 1.0)
            b = cv2.GaussianBlur(cv2.resize(refined[i, oy:oy + T, ox:ox + T].astype(np.float32), (res, res),
                                            interpolation=cv2.INTER_AREA), (0, 0), 1.0)
            return float(np.abs(a - b)[ell].mean())

        tiles = _tiles(lay)
        d = {}
        for (k, i), val in zip(tiles, pmap(one, tiles, workers)):
            t, j = lay["slots"][k][i]
            arr = d.setdefault(id(t), (t, np.full(len(t["cx"]), np.nan)))[1]
            arr[j] = val
        out = {}
        for key, (t, arr) in d.items():
            ok = np.isfinite(arr)
            if ok.sum() < 8:
                continue
            ratio = arr / np.nanmedian(arr)
            w = np.where(ok, np.clip(1 - (ratio - full) / (zero - full), 0, 1), 1.0)
            out[key] = np.minimum(w, _smooth(w, 1.5))
        return out

    def geometry_lock(frames, refined, lay, smooth_frac=0.03, max_frac=0.04, workers=None):
        T = lay["T"]
        h = max(64, T // 2)
        gx, gy = np.meshgrid(np.arange(T, dtype=np.float32), np.arange(T, dtype=np.float32))
        out = refined.copy()

        def grey(x):
            g = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            return cv2.GaussianBlur(cv2.resize(g, (h, h), interpolation=cv2.INTER_AREA), (0, 0), 1.5)

        def one(ki):
            k, i = ki
            ox, oy = lay["origin"][k]
            t, j = lay["slots"][k][i]
            M = _crop_matrix(t["cx"][j], t["cy"][j], t["side"][j], T)
            orig = cv2.warpAffine(frames[i], M, (T, T), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
            tile = refined[i, oy:oy + T, ox:ox + T].astype(np.float32)
            fl = cv2.calcOpticalFlowFarneback(grey(orig), grey(tile), None, 0.5, 4, 15, 3, 7, 1.5, 0)
            fl = np.stack([cv2.GaussianBlur(fl[..., c], (0, 0), smooth_frac * h) for c in range(2)], -1)
            fl = cv2.resize(fl, (T, T), interpolation=cv2.INTER_LINEAR) * (T / h)
            fl = np.clip(fl, -max_frac * T, max_frac * T)
            out[i, oy:oy + T, ox:ox + T] = cv2.remap(tile, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_CUBIC,
                                                     borderMode=cv2.BORDER_REFLECT)

        pmap(one, _tiles(lay), workers)
        return out

    def stitch(frames, refined, lay, mask_scale=1.0, feather=3.0, color_match=1.0, blend=1.0, fade_sigma=4.0,
               fidelity_gate=True, lock_geometry=True, workers=None):
        out = frames.copy()
        n, T, W, H = lay["n"], lay["T"], lay["W"], lay["H"]
        if lock_geometry:
            refined = geometry_lock(frames, refined, lay, workers=workers)
        fid = fidelity(frames, refined, lay, workers=workers) if fidelity_gate else {}

        def sample(ki):
            k, i = ki
            t, j = lay["slots"][k][i]
            ox, oy = lay["origin"][k]
            M = _crop_matrix(t["cx"][j], t["cy"][j], t["side"][j], T)
            orig = cv2.warpAffine(frames[i], M, (T, T), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
            m = _tile_mask(T, t["side"][j], t["fs"][j], mask_scale, 0.5) > 0.5
            return orig[m], refined[i, oy:oy + T, ox:ox + T][m]

        stats = {}

        def gather(tiles):
            for (k, i), (o, r) in zip(tiles, pmap(sample, tiles, workers)):
                st = stats.setdefault(id(lay["slots"][k][i][0]), [[], []])
                st[0].append(o); st[1].append(r)

        gather([(k, i) for k in range(lay["nslots"]) for i in range(0, n, 4) if lay["slots"][k][i] is not None])
        sampled = set(stats)
        gather([(k, i) for k in range(lay["nslots"]) for i in range(n)
                if lay["slots"][k][i] is not None and id(lay["slots"][k][i][0]) not in sampled])
        xf = {}
        for key, (o, r) in stats.items():
            o = np.concatenate(o); r = np.concatenate(r)
            mo, so, mr, sr = o.mean(0), o.std(0) + 1e-4, r.mean(0), r.std(0) + 1e-4
            g = 1 + color_match * (np.clip(so / sr, 0.8, 1.25) - 1)
            xf[key] = (mr, g, mo * color_match + mr * (1 - color_match))
        for i in range(n):  # the fades, in the order the sequential loop first meets each track
            for k in range(lay["nslots"]):
                e = lay["slots"][k][i]
                if e is None or "fade" in e[0]:
                    continue
                t = e[0]
                f = t["found"].astype(np.float64)
                t["fade"] = np.clip(_smooth(f, fade_sigma) * 2, 0, 1)
                if "speed" in t:
                    t["fade"] = t["fade"] * np.clip(1 - (_smooth(t["speed"], 2.0) - 0.3) / 0.2, 0, 1)
                if id(t) in fid:
                    t["fade"] = t["fade"] * fid[id(t)]

        def frame(i):
            parts = []
            for k in range(lay["nslots"]):
                e = lay["slots"][k][i]
                if e is None:
                    continue
                t, j = e
                ox, oy = lay["origin"][k]
                tile = refined[i, oy:oy + T, ox:ox + T].astype(np.float32)
                mr, g, mt = xf[id(t)]
                tile = np.clip((tile - mr) * g + mt, 0, 1)
                side = t["side"][j]
                wgt = blend * t["fade"][j]
                s = max(8, int(round(side)))
                small = cv2.resize(tile, (s, s), interpolation=cv2.INTER_AREA if s < T else cv2.INTER_CUBIC)
                smask = _tile_mask(s, side, t["fs"][j], mask_scale, feather)
                sc = side / s
                tx, ty = t["cx"][j] - side / 2 + 0.5 * sc - 0.5, t["cy"][j] - side / 2 + 0.5 * sc - 0.5
                x0, y0 = max(0, int(np.floor(tx)) - 2), max(0, int(np.floor(ty)) - 2)
                x1, y1 = min(W, int(np.ceil(tx + side)) + 3), min(H, int(np.ceil(ty + side)) + 3)
                if x1 <= x0 or y1 <= y0:
                    continue
                M = np.array([[sc, 0, tx - x0], [0, sc, ty - y0]])
                pt = cv2.warpAffine(small, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                pm = cv2.warpAffine(smask, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * wgt
                parts.append((x0, y0, x1, y1, pt, pm))
            if not parts:
                return
            # accumulate inside the union of the paste boxes only (same sums as on full-frame buffers)
            ux0, uy0 = min(p[0] for p in parts), min(p[1] for p in parts)
            ux1, uy1 = max(p[2] for p in parts), max(p[3] for p in parts)
            acc = np.zeros((uy1 - uy0, ux1 - ux0, 3), np.float32); wsum = np.zeros((uy1 - uy0, ux1 - ux0), np.float32)
            for x0, y0, x1, y1, pt, pm in parts:
                acc[y0 - uy0:y1 - uy0, x0 - ux0:x1 - ux0] += pt * pm[..., None]
                wsum[y0 - uy0:y1 - uy0, x0 - ux0:x1 - ux0] += pm
            if not (wsum > 0).any():
                return
            ys, xs = np.nonzero(wsum > 0)
            ya, yb, xa, xb = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            a = np.clip(wsum[ya:yb, xa:xb], 0, 1)[..., None]
            Y, X = slice(uy0 + ya, uy0 + yb), slice(ux0 + xa, ux0 + xb)
            out[i, Y, X] = out[i, Y, X] * (1 - a) + (acc[ya:yb, xa:xb] / np.maximum(wsum[ya:yb, xa:xb], 1e-6)[..., None]) * a

        pmap(frame, range(n), workers)
        return out

    return stitch, geometry_lock, fidelity
