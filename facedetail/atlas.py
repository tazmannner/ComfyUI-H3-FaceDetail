"""Face atlas: every tracked face of a frame, magnified into its own square tile, packed into ONE canvas so a single
H3 pass re-renders all characters at once (H3's DiT token is 32x32 px: a 45 px face gets ~2 tokens in the wide shot,
~40 in its tile). Optional context: the full frame sits above the tiles (pinned to pass 1) so the tiles see the scene.
Stitch warps each tile back onto its crop box with a feathered ellipse mask over the face."""
import numpy as np
import cv2


def _crop_matrix(cx, cy, side, T):
    """affine mapping tile pixel (u,v) -> frame pixel"""
    s = side / T
    return np.array([[s, 0, cx - side / 2 + 0.5 * s - 0.5], [0, s, cy - side / 2 + 0.5 * s - 0.5]], dtype=np.float64)


def plan(tracks, n, W, H, tile=512, crop_factor=2.6, context=False, min_mag=1.4, max_side_frac=0.95, context_scale=1.0,
         max_mag=12.0, tile_budget=True):
    """assign tracks to slots. Returns dict layout."""
    shots = sorted({t["shot"] for t in tracks})
    by_shot = {}
    for t in tracks:
        side = crop_factor * t["fs"]
        side = np.clip(side, tile / max_mag, max_side_frac * min(W, H))
        act = t.get("active", np.ones(len(side), bool))
        if np.median(tile / side[act]) < min_mag:  # face already big enough on screen
            continue
        t["side"] = side
        by_shot.setdefault(t["shot"], []).append(t)
    for s in by_shot:
        by_shot[s].sort(key=lambda t: float(np.mean(t["cx"])))  # left-to-right, matching "on the left ..." prompts
    nslots = max([len(v) for v in by_shot.values()], default=0)
    if nslots == 0:
        return {"nslots": 0, "n": n, "W": W, "H": H}
    if nslots > 2 and tile_budget:  # keep the face tokens near two full tiles: more faces -> smaller tiles
        tile = max(256, int(tile * (2.0 / nslots) ** 0.5) // 32 * 32)
        for ts in by_shot.values():
            for t in ts:
                t["side"] = np.clip(crop_factor * t["fs"], tile / max_mag, max_side_frac * min(W, H))
    if context:  # frame (optionally downscaled) on top, tiles in a row below it
        Wc, Hc = int(round(W * context_scale / 32)) * 32, int(round(H * context_scale / 32)) * 32
        T = tile
        cw = max(Wc, nslots * T)
        ch = Hc + T
        origin = [((cw - nslots * T) // 2 + k * T, Hc) for k in range(nslots)]
        cols, rows = nslots, 1
    else:
        T = tile
        cols = nslots if nslots <= 3 else (nslots + 1) // 2
        rows = (nslots + cols - 1) // cols
        cw, ch = cols * T, rows * T
        origin = [((k % cols) * T, (k // cols) * T) for k in range(nslots)]
    slots = [[None] * n for _ in range(nslots)]  # slots[k][i] = (track, local index)
    for s, ts in by_shot.items():
        a, b = s
        for k, t in enumerate(ts):
            for i in range(a, b):
                if t.get("active", None) is None or t["active"][i - a]:
                    slots[k][i] = (t, i - a)
    lay = {"nslots": nslots, "n": n, "W": W, "H": H, "T": T, "cw": cw, "ch": ch, "origin": origin, "slots": slots,
           "context": context}
    if context:
        lay["ctx"] = ((cw - Wc) // 2, 0, Wc, Hc)
    return lay


def build(frames, lay, fill=0.5, sharpen=0.0, presmooth=0.0, flatten=0.0):
    """frames (N,H,W,3) float32 -> canvas (N,ch,cw,3), regenerate mask (N,ch,cw)"""
    n, T = lay["n"], lay["T"]
    canvas = np.full((n, lay["ch"], lay["cw"], 3), fill, dtype=np.float32)
    mask = np.zeros((n, lay["ch"], lay["cw"]), dtype=np.float32)
    for i in range(n):
        if lay["context"]:
            x0, y0, Wc, Hc = lay["ctx"]
            f = frames[i] if (Wc, Hc) == (lay["W"], lay["H"]) else cv2.resize(frames[i], (Wc, Hc), interpolation=cv2.INTER_AREA)
            canvas[i, y0:y0 + Hc, x0:x0 + Wc] = f
        for k in range(lay["nslots"]):
            e = lay["slots"][k][i]
            if e is None:
                continue
            t, j = e
            M = _crop_matrix(t["cx"][j], t["cy"][j], t["side"][j], T)
            tile = cv2.warpAffine(frames[i], M, (T, T), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
            if flatten > 0:  # edge-preserving: flattens magnified grain/codec blocks (low contrast), keeps line art
                mag = T / t["side"][j]
                for _ in range(2):
                    tile = cv2.bilateralFilter(tile.astype(np.float32), 0, flatten, 0.6 * mag)
            if presmooth > 0:  # magnified source pixels (grain, codec blocks) become mag-sized blocks the redraw keeps
                tile = cv2.GaussianBlur(tile, (0, 0), presmooth * T / t["side"][j])
            if sharpen > 0:  # magnified tiles are a blurred blob: unsharp at the magnified pixel scale gives crisp edges
                mag = T / t["side"][j]
                tile = tile + sharpen * (tile - cv2.GaussianBlur(tile, (0, 0), 0.6 * mag))
            ox, oy = lay["origin"][k]
            canvas[i, oy:oy + T, ox:ox + T] = np.clip(tile, 0, 1)
            mask[i, oy:oy + T, ox:ox + T] = 1.0
    return canvas, mask


def _tile_mask(T, side, fs, mask_scale, feather):
    """ellipse over the face (tile space), feathered; face is centred in the tile"""
    r = 0.5 * fs * mask_scale * T / side
    m = np.zeros((T, T), np.float32)
    cv2.ellipse(m, (T // 2, int(T // 2 + 0.06 * r)), (int(r), int(r * 1.12)), 0, 0, 360, 1.0, -1)
    k = max(1, int(feather * T / side))
    return cv2.GaussianBlur(m, (0, 0), k)


def stitch(frames, refined, lay, mask_scale=1.5, feather=4.0, color_match=1.0, blend=1.0, fade_sigma=4.0):
    """frames (N,H,W,3), refined canvas (N,ch,cw,3) -> (N,H,W,3)"""
    out = frames.copy()
    n, T, W, H = lay["n"], lay["T"], lay["W"], lay["H"]
    # per track colour transform (one per shot: no flicker), from the original vs refined tiles inside the mask
    stats = {}
    for k in range(lay["nslots"]):
        ox, oy = lay["origin"][k]
        for i in range(0, n, 4):  # colour stats: every 4th frame is plenty for one transform per shot
            e = lay["slots"][k][i]
            if e is None:
                continue
            t, j = e
            M = _crop_matrix(t["cx"][j], t["cy"][j], t["side"][j], T)
            orig = cv2.warpAffine(frames[i], M, (T, T), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
            m = _tile_mask(T, t["side"][j], t["fs"][j], mask_scale, 0.5) > 0.5
            st = stats.setdefault(id(t), [[], []])
            st[0].append(orig[m]); st[1].append(refined[i, oy:oy + T, ox:ox + T][m])
    xf = {}
    for key, (o, r) in stats.items():
        o = np.concatenate(o); r = np.concatenate(r)
        mo, so, mr, sr = o.mean(0), o.std(0) + 1e-4, r.mean(0), r.std(0) + 1e-4
        g = 1 + color_match * (np.clip(so / sr, 0.8, 1.25) - 1)
        xf[key] = (mr, g, mo * color_match + mr * (1 - color_match))
    for i in range(n):
        acc = np.zeros((H, W, 3), np.float32); wsum = np.zeros((H, W), np.float32)
        touched = False
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
            if "fade" not in t:  # fade out where the face has been lost for a while
                f = t["found"].astype(np.float64)
                from .track import _smooth
                t["fade"] = np.clip(_smooth(f, fade_sigma) * 2, 0, 1)
            wgt = blend * t["fade"][j]
            s = max(8, int(round(side)))
            small = cv2.resize(tile, (s, s), interpolation=cv2.INTER_AREA if s < T else cv2.INTER_CUBIC)
            smask = _tile_mask(s, side, t["fs"][j], mask_scale, feather)  # built at on-screen size: cheap blur
            sc = side / s
            tx, ty = t["cx"][j] - side / 2 + 0.5 * sc - 0.5, t["cy"][j] - side / 2 + 0.5 * sc - 0.5
            # warp only into the crop's bounding box (the rest of the frame is untouched)
            x0, y0 = max(0, int(np.floor(tx)) - 2), max(0, int(np.floor(ty)) - 2)
            x1, y1 = min(W, int(np.ceil(tx + side)) + 3), min(H, int(np.ceil(ty + side)) + 3)
            if x1 <= x0 or y1 <= y0:
                continue
            M = np.array([[sc, 0, tx - x0], [0, sc, ty - y0]])
            pt = cv2.warpAffine(small, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            pm = cv2.warpAffine(smask, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) * wgt
            acc[y0:y1, x0:x1] += pt * pm[..., None]; wsum[y0:y1, x0:x1] += pm
            touched = True
        if not touched:
            continue
        ys, xs = np.nonzero(wsum > 0)
        ya, yb, xa, xb = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        a = np.clip(wsum[ya:yb, xa:xb], 0, 1)[..., None]
        out[i, ya:yb, xa:xb] = out[i, ya:yb, xa:xb] * (1 - a) + (acc[ya:yb, xa:xb] / np.maximum(wsum[ya:yb, xa:xb], 1e-6)[..., None]) * a
    return out


def preview(frames, lay):
    out = (np.clip(frames, 0, 1) * 255).astype(np.uint8).copy()
    cols = [(255, 64, 64), (64, 255, 64), (64, 160, 255), (255, 220, 0)]
    for k in range(lay["nslots"]):
        for i in range(lay["n"]):
            e = lay["slots"][k][i]
            if e is None:
                continue
            t, j = e
            s, cx, cy, fs = t["side"][j], t["cx"][j], t["cy"][j], t["fs"][j]
            c = cols[k % 4] if t["found"][j] else (128, 128, 128)
            cv2.rectangle(out[i], (int(cx - s / 2), int(cy - s / 2)), (int(cx + s / 2), int(cy + s / 2)), c, 1)
            cv2.rectangle(out[i], (int(cx - fs / 2), int(cy - fs / 2)), (int(cx + fs / 2), int(cy + fs / 2)), c, 1)
    return out.astype(np.float32) / 255
