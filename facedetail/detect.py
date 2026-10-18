"""Anime face detection: deepghs anime_face_detection (YOLOv8, ONNX, CPU onnxruntime). No pickles, no ultralytics."""
import os
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "anime_face_detect_v1.4_s")
_sess = None


def session():
    global _sess
    if _sess is None:
        import onnxruntime as ort
        path = os.path.join(MODEL_DIR, "model.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"anime face detector missing: put model.onnx from https://huggingface.co/deepghs/anime_face_detection "
                f"(folder face_detect_v1.4_s) at {path}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        _sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    return _sess


def _nms(boxes, scores, iou=0.5):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        o = inter / (a[i] + a[order[1:]] - inter + 1e-9)
        order = order[1:][o < iou]
    return keep


def detect(frame, size=1024, conf=0.25, iou=0.5):
    """frame: HxWx3 float [0,1] or uint8. Returns [(x1,y1,x2,y2,score)] in frame pixels."""
    f = np.asarray(frame)
    if f.dtype != np.uint8:
        f = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    H, W = f.shape[:2]
    r = size / max(H, W)
    nh, nw = int(round(H * r)), int(round(W * r))
    import cv2
    img = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    ph, pw = (32 - nh % 32) % 32, (32 - nw % 32) % 32
    img = np.pad(img, ((0, ph), (0, pw), (0, 0)), constant_values=114)
    x = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = session().run(None, {"images": x})[0][0]  # (5, anchors)
    if out.shape[0] != 5:
        out = out.T
    cx, cy, bw, bh, sc = out
    m = sc > conf
    if not m.any():
        return []
    b = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)[m] / r
    s = sc[m]
    k = _nms(b, s, iou)
    b = b[k]; s = s[k]
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W); b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
    return [tuple(map(float, bb)) + (float(ss),) for bb, ss in zip(b, s)]
