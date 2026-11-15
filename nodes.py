import numpy as np
import torch
from .facedetail import detect, track, atlas, compact

CAT = "H3 FaceDetail"


class H3FaceAtlas:
    DESCRIPTION = ("Wide shots: H3 draws each 32x32 px DiT token as one unit, so a small face gets only a few tokens and "
                   "comes out mushy. This finds every anime face (deepghs ONNX detector, CPU), tracks it per shot, and packs "
                   "each face magnified into its own tile of ONE canvas, so a single H3 pass redraws all characters with "
                   "~20x more tokens per face. context=True puts the full frame above the tiles (pinned to pass 1 via the "
                   "regen mask) so the faces are redrawn while seeing the whole scene.")
    RETURN_TYPES = ("IMAGE", "MASK", "H3_FACE_ATLAS", "IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("canvas", "regen_mask", "atlas", "preview", "width", "height", "report")
    FUNCTION = "run"
    CATEGORY = CAT

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "tile": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 32}),
            "crop_factor": ("FLOAT", {"default": 2.6, "min": 1.5, "max": 5.0, "step": 0.1,
                            "tooltip": "crop side / face size: the face fills ~1/crop_factor of its tile"}),
            "context": ("BOOLEAN", {"default": False}),
            "min_magnification": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 8.0, "step": 0.1,
                                  "tooltip": "faces already this big on screen are left alone"}),
            "max_faces": ("INT", {"default": 4, "min": 1, "max": 8}),
            "confidence": ("FLOAT", {"default": 0.2, "min": 0.05, "max": 0.95, "step": 0.05}),
            "detect_size": ("INT", {"default": 1024, "min": 320, "max": 2048, "step": 32}),
        }, "optional": {
            "context_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 1.0, "step": 0.05,
                              "tooltip": "size of the pinned full-frame context panel (context=True); 0.5 = a quarter of the tokens"}),
            "flatten": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                        "tooltip": "edge-preserving flatten of the magnified tile (bilateral colour sigma): removes the magnified grain grid, keeps line art"}),
            "presmooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                          "tooltip": "gaussian blur of the magnified tile, in source pixels: removes the magnified pixel grid"}),
            "sharpen": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1,
                        "tooltip": "unsharp mask on the magnified tiles before the redraw (scaled to the magnification)"}),
            "make_preview": ("BOOLEAN", {"default": True,
                             "tooltip": "off: the `preview` output is only its first frame. Turn it off when nothing shows "
                                        "the preview: a full-size copy of the clip otherwise stays in ComfyUI's cache"}),
        }}

    def run(self, images, tile, crop_factor, context, min_magnification, max_faces, confidence, detect_size, context_scale=1.0, sharpen=0.0, presmooth=0.0, flatten=0.0, make_preview=True):
        fr = images.cpu().numpy().astype(np.float32, copy=False)   # only read here: no second copy of the clip
        n, H, W = fr.shape[:3]
        # detect every 2nd frame (tracks are interpolated + smoothed anyway); the odd frames get no detections
        dets = [detect.detect(f, size=detect_size, conf=confidence) if i % 2 == 0 or i == n - 1 else []
                for i, f in enumerate(fr)]
        starts = track.shot_cuts(fr)
        tr = track.build_tracks(dets, starts, n, W, H, max_faces=max_faces)
        lay = atlas.plan(tr, n, W, H, tile=tile, crop_factor=crop_factor, context=context, min_mag=min_magnification,
                          context_scale=context_scale)
        rep = [f"{n} frames {W}x{H}, shots at {starts}, {len(tr)} tracks, {lay['nslots']} slots"]
        for t in tr:
            mag = np.median(lay.get('T', tile) / t['side'][t['active']]) if 'side' in t else 0
            rep.append(f"  shot {t['shot']}: face ~{np.median(t['fs']):.0f}px found {t['found'].mean()*100:.0f}% "
                       f"frames {np.flatnonzero(t['active'])[0]+t['shot'][0]}-{np.flatnonzero(t['active'])[-1]+t['shot'][0]} x~{np.mean(t['cx']):.0f} mag {mag:.1f}x {'(skipped: big)' if 'side' not in t else ''}")
        if lay["nslots"] == 0:
            empty = torch.zeros((n, 64, 64, 3))
            return (empty, torch.zeros((n, 64, 64)), lay, images, 64, 64, "\n".join(rep + ["no faces to fix"]))
        canvas, mask = atlas.build(fr, lay, sharpen=sharpen, presmooth=presmooth, flatten=flatten)
        rep.append(f"canvas {lay['cw']}x{lay['ch']} tile {lay['T']}")
        print("[H3FaceAtlas]\n" + "\n".join(rep))
        prev = atlas.preview(fr, lay) if make_preview else atlas.preview(fr[:1], dict(lay, n=1))
        return (torch.from_numpy(canvas), torch.from_numpy(mask), lay, torch.from_numpy(prev),
                lay["cw"], lay["ch"], "\n".join(rep))


class H3FaceAtlasStitch:
    DESCRIPTION = "Warp the redrawn face tiles back onto the original frames (feathered ellipse over each face, one colour match per shot)."
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",), "refined": ("IMAGE",), "atlas": ("H3_FACE_ATLAS",),
            "mask_scale": ("FLOAT", {"default": 1.0, "min": 0.8, "max": 3.0, "step": 0.05,
                           "tooltip": "paste ellipse diameter / detected face size. 1.0 = the face only: hair outline and "
                                      "neckline stay pass 1's (steady); bigger pastes more of the redraw and its edges"}),
            "feather": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 20.0, "step": 0.5, "tooltip": "source pixels"}),
            "color_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }, "optional": {
            "geometry_lock": ("BOOLEAN", {"default": True, "tooltip": "warp each redrawn face onto pass 1's line positions "
                              "(smoothed optical flow): stops the outlines shimmering against the untouched pixels around "
                              "the patch, keeps the redraw's detail"}),
            "fidelity_gate": ("BOOLEAN", {"default": True, "tooltip": "keep the original on frames where the redraw "
                              "reinvented the head (spins, back of the head, motion-blurred turns) instead of refining it"}),
        }}

    def run(self, images, refined, atlas, mask_scale, feather, color_match, blend, fidelity_gate=True, geometry_lock=True):
        if atlas["nslots"] == 0:
            return (images,)
        fr = images.cpu().numpy().astype(np.float32, copy=False)   # both are only read: the stitch writes a new array
        rf = refined.cpu().numpy().astype(np.float32, copy=False)
        if rf.shape[0] != fr.shape[0]:
            raise ValueError(f"refined has {rf.shape[0]} frames, images {fr.shape[0]}")
        if rf.shape[1:3] != (atlas["ch"], atlas["cw"]):
            raise ValueError(f"refined is {rf.shape[2]}x{rf.shape[1]}, atlas canvas {atlas['cw']}x{atlas['ch']}")
        out = atlas_mod.stitch(fr, rf, atlas, mask_scale=mask_scale, feather=feather, color_match=color_match, blend=blend,
                                fidelity_gate=fidelity_gate, lock_geometry=geometry_lock)
        return (torch.from_numpy(out),)


class H3FaceAtlasSigmas:
    DESCRIPTION = ("Redraw schedule for the face atlas. Smaller faces are magnified more and their tiles start as a soft "
                   "blob, so they need a deeper redraw: the start sigma is interpolated from the faces' median "
                   "magnification (sigma_at_3x .. sigma_at_5x .. sigma_at_10x), then `steps` euler steps on H3's shift-12 curve.")
    RETURN_TYPES = ("SIGMAS", "FLOAT", "STRING")
    RETURN_NAMES = ("sigmas", "start_sigma", "report")
    FUNCTION = "run"
    CATEGORY = CAT

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "atlas": ("H3_FACE_ATLAS",),
            "steps": ("INT", {"default": 4, "min": 1, "max": 30}),
            "sigma_at_3x": ("FLOAT", {"default": 0.70, "min": 0.3, "max": 1.0, "step": 0.01,
                            "tooltip": "faces that are already fairly big get a light touch-up (keeps their acting)"}),
            "sigma_at_5x": ("FLOAT", {"default": 0.80, "min": 0.3, "max": 1.0, "step": 0.01}),
            "sigma_at_10x": ("FLOAT", {"default": 0.95, "min": 0.3, "max": 1.0, "step": 0.01}),
            "shift": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 30.0, "step": 0.5}),
        }}

    def run(self, atlas, steps, sigma_at_5x, sigma_at_10x, shift, sigma_at_3x=0.70):
        mags = []
        for k in range(atlas.get("nslots", 0)):
            for e in atlas["slots"][k]:
                if e is not None:
                    t, j = e
                    mags.append(atlas["T"] / t["side"][j])
        mag = float(np.median(mags)) if mags else 5.0
        s0 = float(np.interp(mag, [3.0, 5.0, 10.0], [sigma_at_3x, sigma_at_5x, sigma_at_10x]))
        t0 = s0 / (shift - (shift - 1) * s0)
        ts = [t0 * (1 - k / steps) for k in range(steps + 1)]
        sig = torch.tensor([shift * t / (1 + (shift - 1) * t) for t in ts], dtype=torch.float32)
        return (sig, s0, f"median magnification {mag:.1f}x -> start sigma {s0:.3f}: " + ", ".join(f"{v:.3f}" for v in sig.tolist()))


atlas_mod = atlas
class H3FaceAtlasCompact:
    DESCRIPTION = ("Speed: drops the canvas frames that hold no face tile (close-ups, shots without small faces) before "
                   "the redraw, so the encode, the H3 pass and the decode only run on frames with a face to fix. Wire "
                   "canvas / regen_mask / audio from here into the redraw, `length` into its conditioning, and decode "
                   "with H3 Face Atlas Decode, which puts the frames back in place. Clips with small faces in every "
                   "frame pass through unchanged.")
    RETURN_TYPES = ("IMAGE", "MASK", "AUDIO", "H3_FACE_COMPACT", "INT", "STRING")
    RETURN_NAMES = ("canvas", "regen_mask", "audio", "compact", "length", "report")
    FUNCTION = "run"
    CATEGORY = CAT

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "canvas": ("IMAGE",), "regen_mask": ("MASK",), "atlas": ("H3_FACE_ATLAS",),
            "min_gap": ("INT", {"default": 17, "min": 0, "max": 240,
                        "tooltip": "gaps without a face tile up to this many frames are kept (a jump cut in the canvas "
                                   "for a few frames saved isn't worth it)"}),
        }, "optional": {
            "audio": ("AUDIO", {"tooltip": "the clip's audio, when the redraw pins it: cut to the same frames (only the "
                                           "redraw hears this copy; the clip's soundtrack is not touched)"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
        }}

    def run(self, canvas, regen_mask, atlas, min_gap, audio=None, fps=24.0):
        n = canvas.shape[0]
        pl = compact.plan(atlas, min_gap) if atlas.get("nslots", 0) and atlas.get("n") == n else None
        if pl is None:
            rep = f"all {n} frames kept (nothing to drop)"
            print("[H3FaceAtlasCompact] " + rep)
            return (canvas, regen_mask, audio, {"n": n, "idx": None}, n, rep)
        idx, k = pl
        ti = torch.from_numpy(idx)
        mask = regen_mask[ti] if regen_mask.shape[0] == n else regen_mask
        if mask.shape[0] == len(idx) and len(idx) > k:
            mask = mask.clone()
            mask[k:] = 0                      # the padding frames are frozen copies, nothing to redraw
        out_audio = audio
        if audio is not None:
            w, sr = audio["waveform"], audio["sample_rate"]
            spf = sr / fps
            parts = []
            for j, i in enumerate(idx):
                a, b = int(round(i * spf)), int(round((i + 1) * spf))
                seg = w[..., a:b]
                want = int(round((j + 1) * spf)) - int(round(j * spf))
                if j >= k:
                    seg = torch.zeros_like(w[..., :want])   # silence under the padding frames
                elif seg.shape[-1] < want:
                    seg = torch.nn.functional.pad(seg, (0, want - seg.shape[-1]))
                parts.append(seg[..., :want])
            out_audio = dict(audio, waveform=torch.cat(parts, dim=-1))
        rep = f"redraw {k} of {n} frames (+{len(idx) - k} padding to {len(idx)}): {n - k} frames hold no face tile"
        print("[H3FaceAtlasCompact] " + rep)
        return (canvas[ti], mask, out_audio, {"n": n, "idx": idx[:k].tolist()}, len(idx), rep)


class H3FaceAtlasDecode:
    DESCRIPTION = ("VAE Decode for the face canvas after H3 Face Atlas Compact: decodes the redrawn frames and puts each "
                   "back at its place in the clip (the skipped frames stay empty; the stitch never reads them).")
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CAT

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",), "compact": ("H3_FACE_COMPACT",)}}

    def run(self, samples, vae, compact):
        import nodes as comfy_nodes
        dec = comfy_nodes.VAEDecode().decode(vae, samples)[0]
        if compact.get("idx") is None:
            return (dec,)
        idx = torch.tensor(compact["idx"])
        if dec.shape[0] < len(idx):
            raise ValueError(f"decoded {dec.shape[0]} frames, the compact map has {len(idx)}")
        out = torch.zeros((compact["n"],) + tuple(dec.shape[1:]), dtype=dec.dtype)
        out[idx] = dec[:len(idx)]
        return (out,)


NODE_CLASS_MAPPINGS = {"H3FaceAtlas": H3FaceAtlas, "H3FaceAtlasStitch": H3FaceAtlasStitch, "H3FaceAtlasSigmas": H3FaceAtlasSigmas,
                       "H3FaceAtlasCompact": H3FaceAtlasCompact, "H3FaceAtlasDecode": H3FaceAtlasDecode}
NODE_DISPLAY_NAME_MAPPINGS = {"H3FaceAtlas": "H3 Face Atlas (wide-shot faces)", "H3FaceAtlasStitch": "H3 Face Atlas Stitch",
                              "H3FaceAtlasSigmas": "H3 Face Atlas Sigmas (by face size)",
                              "H3FaceAtlasCompact": "H3 Face Atlas Compact (skip frames without faces)",
                              "H3FaceAtlasDecode": "H3 Face Atlas Decode (after Compact)"}
