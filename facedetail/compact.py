"""Time compaction: the face canvas is only worth redrawing on frames that hold a face tile. Close-ups and shots
without small faces have none, yet the canvas pass used to run over the whole clip."""
import numpy as np


def valid_length(k):
    """H3 video lengths are 17 m + 5 frames (17-frame VAE clips of 5 latent frames, plus the closing 5)"""
    return max(22, int(np.ceil((k - 5) / 17.0)) * 17 + 5)


def plan(lay, min_gap=17):
    """frames of the canvas to redraw: those with a face tile, short gaps between them (< min_gap) kept as well, then
    the last kept frame repeated up to a valid H3 length. Returns (frame indices incl. the padding, frames kept) or
    None when nothing can be dropped."""
    n = lay["n"]
    keep = np.zeros(n, bool)
    for s in lay.get("slots", []):
        keep |= np.array([x is not None for x in s], bool)
    if not keep.any():
        return None
    idx = np.flatnonzero(keep)
    for a, b in zip(idx[:-1], idx[1:]):       # a short gap costs less than the jump cut it would add
        if 1 < b - a <= min_gap:
            keep[a:b] = True
    idx = np.flatnonzero(keep)
    k = len(idx)
    length = valid_length(k)
    if length >= n:
        return None
    return np.concatenate([idx, np.full(length - k, idx[-1])]).astype(int), k
