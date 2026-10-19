# ComfyUI-H3-FaceDetail

**Fixes mushy faces in MiniMax H3 wide shots: one extra pass redraws every small face at once, while it sees the whole scene.**

Built for anime and tested on Re:Zero, Cyberpunk Edgerunners, Teen Titans and Golden Boy style LoRAs and on base-model anime. It also works on live action: the anime face detector finds real faces too.

## Why H3 wide-shot faces go mushy

H3's video VAE compresses 16x spatially, and the DiT then patches 2x2 latents, so **every DiT token covers 32x32 pixels**.
At 1024x576 the model draws on a 32x18 grid. A full-body wide shot puts a 35 px face on that grid, which is about 1.5x1.5
tokens: the eyes, brows and mouth have nowhere to go, so the model paints a blob. Rendering at 1 MP barely helps,
because the face stays the same small share of the frame (~2 tokens at 1376x768 too).

So the fix has to give faces more tokens. Re-rendering the whole frame bigger costs the square of the token count
(4x tokens = ~10x attention).

## What this pack does

1. **H3 Face Atlas** finds every anime face. It uses the ONNX anime face detector `deepghs/anime_face_detection`
   v1.4 s on CPU, with no pickles and no ultralytics. It splits the clip into shots at hard cuts and smooths one
   track per character per shot. Each face is then magnified into its own 512 px tile of **one canvas**: about 60
   tokens per face instead of 2, and every character goes through a single pass.
   With `context` on (the default in the workflows), the whole frame sits above the tiles at half size. Its regen
   mask is 0, so it is pinned and never redrawn, and the face tiles attend to it. The faces are redrawn while seeing
   the actual scene, its lighting and its effects. Tiles-only redraws pasted stray effects (fireflies) onto faces and
   boiled more frame to frame; with context they stayed as stable as pass 1.
2. The canvas goes through **H3 V2V Init** (MAINodes). The regen mask covers only the tiles, and pass 1's audio is
   **pinned**, so mouths keep following the dialogue. It is then sampled with the DMAD turbo LoRA for 4 steps.
3. **H3 Face Atlas Sigmas** picks how deep to redraw from the faces' magnification. A 55 px face (~3x) gets a light
   touch-up from sigma 0.70, which keeps its mouth timing. A 35 px face (~5x) starts at 0.80. A 20 px face (~10x)
   starts at 0.95, because its tile begins as a soft blob that a shallow redraw keeps (it comes out soft or blocky).
4. **H3 Face Atlas Stitch** warps each redrawn face back onto its sub-pixel crop box under a feathered ellipse. It
   applies one colour match per shot, which avoids flicker, and fades out where the face was lost.
   A **fidelity gate** compares each redrawn face with the original at on-screen size. Where the redraw reinvented
   the head instead of refining it, that frame keeps the original. Examples: a spin, the back of a head, a
   motion-blurred turn. "Reinvented" means more than ~2x the track's usual difference.
   On fast heads the crop follows the detections closely instead of a smoothed path, so it never lands beside the face.
   A **geometry lock** warps each redrawn face onto pass 1's line positions before pasting. It uses heavily smoothed
   optical flow from the original crop, so only regional drift is corrected and the new detail stays. Without it the
   redraw places the jaw, collar and hair edge about a pixel differently every frame. Next to pass 1's steady pixels
   outside the mask, that reads as the patch shifting. The paste mask covers only the face (`mask_scale` 1.0), so hair
   outline and neckline stay pass 1's. Neck-band shimmer on a walking two-shot: 1.06 → 0.22 (lossless), which is below
   h264's own noise.

Faces that are already big on screen (close-ups) are skipped and pass through untouched. A clip with no small faces
costs only the detection.

## Cost (RTX 5090, 1024x576, 124 frames)

| | pass 1 (20 steps) | face pass | of which sampling |
|---|---|---|---|
| 1024x576, 2 faces | 84 s | ~45 s | 24 s |
| 1024x576, 3–4 faces (smaller tiles) | 79–83 s | ~60 s | ~30 s |
| 1344x768 (1 MP), 2 faces | 166 s | ~65 s | 28 s |

The rest is detection (~5 s, CPU, every 2nd frame), VAE encode of the canvas (~10-15 s), decode (~5 s) and stitch (~2 s).
In the full Fast De-rope + face fix + Anime Cadence graph a 5 s clip took 141-159 s, against ~84 s for a plain render.

## Nodes

| node | key inputs |
|---|---|
| H3 Face Atlas (wide-shot faces) | `tile` 512, `crop_factor` 2.6 (face ~40% of its tile), `context` + `context_scale` 0.5, `min_magnification` 1.4 (skip big faces), `confidence` 0.2 |
| H3 Face Atlas Sigmas (by face size) | `steps` 4, `sigma_at_3x` 0.70, `sigma_at_5x` 0.80, `sigma_at_10x` 0.95 |
| H3 Face Atlas Stitch | `mask_scale` 1.0 (face only), `feather` 3 px, `color_match` 1.0, `geometry_lock` on, `fidelity_gate` on |

Wiring: `canvas` → VAEEncode → H3V2VInit (`mask` = `regen_mask`, `time_varying` on, `audio_latent` = VAEEncodeAudio of
the render's audio, audio_mode *pin*) → SamplerCustomAdvanced (`sigmas` from H3 Face Atlas Sigmas; the
MiniMaxH3ImageToVideo cond uses the atlas `width` / `height` and the clip length) → VAEDecode → Stitch.
Use the **same prompt and style LoRA** as the render.

Example workflows:
- `example_workflows/h3_wide_face_fix.json` fixes any existing H3 video.
- `h3_fast_derope_face.json` runs t2v → Fast De-rope → face fix → Anime Cadence.
- `h3_fast_derope_face_liveaction.json` is the same without the 2s drawings and Anime Cadence.

## Requirements

- `onnxruntime` and `opencv` (both already in a normal ComfyUI install)
- ComfyUI-MAINodes (H3V2VInit)
- for the two Fast De-rope example workflows: ComfyUI-H3-FastDerope at cf02e34 or later (Motion Gate `static_images`)
- the detector: `models/anime_face_detect_v1.4_s/model.onnx` from
  https://huggingface.co/deepghs/anime_face_detection (folder `face_detect_v1.4_s`, sha256 `403b5bc9…6891483`)

## What did not work (so you don't have to try it)

- **Tiles without the context panel**: stray scene effects (fireflies) were pasted onto faces.
- **Blurring the tile before the redraw**: the model keeps the blur, and faces come out soft. **Sharpening it** amplifies
  the magnified pixel grid. **Edge-preserving flattening** changes nothing: the blocky look of tiny faces comes from a
  redraw that is too shallow, not from the input. A deeper sigma for small faces fixes it.
- **Encoding the canvas with taeh3** (10 s faster): thin streaks across faces. Use the real VAE.
- **6 steps** instead of 4 made no visible difference.

## Limitations

- Faces under ~15 px are capped at 12x magnification. There is little a 15 px face can show at the output resolution.
- Faces seen from behind, or not detected, are left as they are. Faces on billboards and screens are faces too: they
  get a slot and a light redraw (harmless), and with more than 2 slots every tile shrinks. Lower `max_faces` or raise
  `min_magnification` if a big background face steals a slot.
- More than 2 faces: tiles shrink to keep the token count near two full tiles (3 faces → 416 px, 4 → 352 px), so the
  face pass costs about the same with 3–4 characters as with 2.
- Mouths keep pass 1's open/closed timing (checked frame by frame), but the exact mouth shape is redrawn. At
  sigma 0.8 or more a long held-open mouth can come back flapping open and shut. Faces of ~55 px therefore get the
  lighter 0.70 redraw.
- The redrawn face is drawn at 512 px and downscaled into the frame. At 576p output it is as sharp as the frame
  allows, not sharper.
