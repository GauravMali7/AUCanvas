# draw_utils.py
import cv2
import math
import numpy as np
from typing import Sequence, Optional
from au_canvas.paint_config import ALL_MUSCLE_PAIRS
from au_canvas.au_config import INDEX_LIST, THRESHOLDS, AU_INDEX, AU_NAMES
from au_canvas.paint_config import ALL_MUSCLE_PAIRS, AU_TO_MUSCLES, MUSCLE_BY_NAME
from au_canvas.au_config import INDEX_LIST, THRESHOLDS, AU_INDEX, AU_NAMES, CODE_TO_IDX, THRESH_BY_CODE

# ---------- geometry helpers ----------
def _to_pts(face_norm_landmarks, idxs, w, h):
    pts = np.array([[face_norm_landmarks[i][0]*w, face_norm_landmarks[i][1]*h] for i in idxs], dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:,  1], 0, h - 1)
    return pts

def _catmull_rom_segment(P0, P1, P2, P3, n=12):
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    t2, t3 = t*t, t*t*t
    a = -P0 + 3*P1 - 3*P2 + P3
    b =  2*P0 - 5*P1 + 4*P2 - P3
    c = -P0 + P2
    d =  2*P1
    return 0.5*(a*t3 + b*t2 + c*t + d)

def _catmull_rom_closed(pts, samples_per_seg=12):
    pts = np.asarray(pts, dtype=np.float32)
    n = len(pts)
    if n < 3:
        return pts
    out = []
    for i in range(n):
        P0 = pts[(i-1) % n]
        P1 = pts[i % n]
        P2 = pts[(i+1) % n]
        P3 = pts[(i+2) % n]
        seg = _catmull_rom_segment(P0, P1, P2, P3, n=samples_per_seg)
        out.append(seg[:-1])
    return np.vstack(out)

# ---------- muscle painter ----------
def draw_muscle_patch(
    img_bgr,
    face_norm_landmarks,
    upper_indices: Sequence[int],
    lower_indices: Sequence[int],
    *,
    color=(0, 0, 255),
    core_alpha=0.65,
    edge_alpha=0.38,
    inner_width_px=10,
    outer_width_px=18,
    samples_per_seg=14,
    feather_extra_px=4,
    cut_upper_indices: Optional[Sequence[int]] = None,
    cut_lower_indices: Optional[Sequence[int]] = None,
    cut_type="poly",
    cut_feather_px=3.0,
    cut_dilate_px=0.0,
    cut_use_catmull=True,
    cut_samples_per_seg=12,
    border_px=0.8,
    border_color=(128,128,128),
    border_alpha=0.20,
    border_shadow_px=0.1,
    border_shadow_color=(0,0,0),
    border_scale_by_size=True,
    debug_outline=False
):
    if not face_norm_landmarks:
        return img_bgr
    h, w = img_bgr.shape[:2]

    up = _to_pts(face_norm_landmarks, upper_indices, w, h)
    lo = _to_pts(face_norm_landmarks, lower_indices, w, h)[::-1]
    ring = np.vstack([up, lo])
    ring_smooth = _catmull_rom_closed(ring, samples_per_seg=samples_per_seg)

    x_min, y_min = np.floor(ring_smooth.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(ring_smooth.max(axis=0)).astype(int)
    margin = int(max(6, inner_width_px + outer_width_px + feather_extra_px + 6))
    x0 = max(0, x_min - margin);  y0 = max(0, y_min - margin)
    x1 = min(w, x_max + margin);  y1 = min(h, y_max + margin)
    if x1 <= x0 or y1 <= y0:
        return img_bgr

    poly_roi = (ring_smooth - np.array([x0, y0], dtype=np.float32)).astype(np.int32)
    roi = img_bgr[y0:y1, x0:x1]
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_roi], 255, lineType=cv2.LINE_AA)

    dist_in  = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    dist_out = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    alpha_map = np.zeros_like(dist_in, dtype=np.float32)

    if inner_width_px > 0:
        t_in = np.clip(dist_in / float(max(1e-6, inner_width_px)), 0.0, 1.0)
        alpha_inside = edge_alpha + (core_alpha - edge_alpha) * t_in
    else:
        alpha_inside = np.full_like(dist_in, core_alpha, dtype=np.float32)
    alpha_map[mask > 0] = alpha_inside[mask > 0]

    if outer_width_px > 0:
        t_out = np.clip(1.0 - dist_out / float(max(1e-6, outer_width_px)), 0.0, 1.0)
        alpha_outside = edge_alpha * t_out
        outside = (mask == 0)
        alpha_map[outside] = np.maximum(alpha_map[outside], alpha_outside[outside])

    if feather_extra_px and feather_extra_px > 0:
        k = 2 * int(3 * feather_extra_px) + 1
        alpha_map = cv2.GaussianBlur(alpha_map, (k, k), feather_extra_px)

    # ----- cutout / hole -----
    if (cut_upper_indices is not None) and (cut_lower_indices is not None):
        up_cut = _to_pts(face_norm_landmarks, cut_upper_indices, w, h)
        lo_cut = _to_pts(face_norm_landmarks, cut_lower_indices, w, h)[::-1]
        ring_cut = np.vstack([up_cut, lo_cut])
        cut_poly_roi = (ring_cut - np.array([x0, y0], dtype=np.float32))
        cut_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        if cut_type == "poly":
            if cut_use_catmull and len(cut_poly_roi) >= 3:
                cut_poly_roi = _catmull_rom_closed(cut_poly_roi, samples_per_seg=cut_samples_per_seg)
            cv2.fillPoly(cut_mask, [cut_poly_roi.astype(np.int32)], 255, lineType=cv2.LINE_AA)
        else:
            if len(cut_poly_roi) >= 5:
                ellipse = cv2.fitEllipse(cut_poly_roi.astype(np.float32).reshape(-1, 1, 2))
                (cx, cy), (diam_a, diam_b), ang = ellipse
                axes = (max(1, int(diam_a/2.0)), max(1, int(diam_b/2.0)))
                cv2.ellipse(cut_mask, (int(cx), int(cy)), axes, float(ang), 0, 360, 255, -1, cv2.LINE_AA)

        if cut_dilate_px and cut_dilate_px > 0:
            k = 2 * int(cut_dilate_px) + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            cut_mask = cv2.dilate(cut_mask, kernel)

        if cut_feather_px and cut_feather_px > 0:
            k = 2 * int(3 * cut_feather_px) + 1
            soft = cv2.GaussianBlur(cut_mask.astype(np.float32)/255.0, (k, k), cut_feather_px)
            alpha_map *= (1.0 - soft)
        else:
            alpha_map[cut_mask > 0] = 0.0

    overlay = np.empty_like(roi); overlay[:] = np.array(color, dtype=np.uint8)
    out_roi = (overlay.astype(np.float32) * alpha_map[..., None] +
               roi.astype(np.float32) * (1.0 - alpha_map[..., None])).astype(np.uint8)

    if border_px and border_px > 0:
        diag = float(np.hypot(x_max - x_min, y_max - y_min))
        t = int(max(1, border_px if not border_scale_by_size else max(border_px, 0.006 * diag)))
        if t > 0:
            stroke_layer = out_roi.copy()
            cv2.polylines(stroke_layer, [poly_roi], True, (128,128,128), t, cv2.LINE_AA)
            out_roi = cv2.addWeighted(stroke_layer, border_alpha, out_roi, 1.0 - border_alpha, 0.0)

    if debug_outline:
        cv2.polylines(out_roi, [poly_roi], True, (0,255,255), 1, cv2.LINE_AA)

    out = img_bgr.copy()
    out[y0:y1, x0:x1] = out_roi
    return out

# ---------- AU sidebar / outer box ----------
def _draw_bar(img, x, y, w, h, p, thr):
    p = float(max(0.0, min(1.0, p)))
    thr = float(max(0.0, min(1.0, thr)))
    # background + border
    cv2.rectangle(img, (x, y), (x+w, y+h), (36,36,36), -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (95,95,95), 2, cv2.LINE_AA)
    # value fill
    fw = int(round((w-4) * p))
    if p < thr:
        color =  (70,200,90)
    else:
        color =  (0,0,255)
    if fw > 0:
        cv2.rectangle(img, (x+2, y+2), (x+2+fw, y+h-2), color, -1)
    # threshold tick
    tx = x + int(round(w * thr))
    cv2.line(img, (tx, y-1), (tx, y+h+1), (0,0,255), 2, cv2.LINE_AA)

def make_info_panel(
    h, w=520, *,
    au_probs=None, playback_fps=0.0, infer_fps=0.0,
    frame_idx=0, total_frames=0, run_time_s=0.0,
    cols=2
):
    """Wider, clearer panel with two-column AU bars and a controls box."""
    panel = np.full((h, w, 3), (24,24,24), dtype=np.uint8)  # dark sidebar

    pad = 16
    y = pad + 6
    font = cv2.FONT_HERSHEY_SIMPLEX

    def put(txt, color=(235,235,235), scale=0.7, th=2, dy=26):
        nonlocal y
        cv2.putText(panel, txt, (pad, y), font, scale, color, th, cv2.LINE_AA)
        y += dy

    # Header
    y += 10
    cv2.putText(panel, "AU Video Viewer", (pad, y), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
    y += 34

    # Stats row
    put(f"Frame: {frame_idx}/{max(0,total_frames-1)}", scale=0.62, th=2, dy=34)
    put(f"Run time: {run_time_s:.2f}s",               scale=0.62, th=2, dy=34)
    put(f"Playback FPS: {playback_fps:.2f}",          scale=0.62, th=2, dy=34)
    put(f"Infer FPS: {infer_fps:.2f}",                scale=0.62, th=2, dy=34)

    # divider
    cv2.line(panel, (pad, y), (w-pad, y), (80,80,80), 1, cv2.LINE_AA)
    y += 10

    # AU section
    if au_probs is not None:
        # Legend
        cv2.putText(panel, "AUs (green=prob, red=activated)", (pad, y+20), font, 1, (210,210,210), 2, cv2.LINE_AA)
        y += 50

        # layout
        col_count = max(1, int(cols))
        col_w = (w - 2*pad) / col_count
        label_w = 340  # space for "AUxx Name"
        bar_gap = 10
        bar_h = 16
        row_h = 40

        rows_per_col = math.ceil(len(INDEX_LIST) / col_count)

        # draw entries by column
        for i, (idx, thr) in enumerate(zip(INDEX_LIST, THRESHOLDS)):
            col = i // rows_per_col
            row = i % rows_per_col
            col_x = int(pad + col * col_w)
            yy = int(y + row * row_h)

            # label
            code = AU_INDEX[idx] if idx < len(AU_INDEX) else str(idx)
            name = AU_NAMES[idx] if idx < len(AU_NAMES) else f"AU{code}"
            label = f"AU{code} {name}"
            if au_probs[idx]<thr:
                color = (220,220,220)
            else:
                color = (0,0,255)
            cv2.putText(panel, label, (col_x, yy + bar_h - 2), font, 0.7, color, 2, cv2.LINE_AA)

            # bar
            bx = col_x + label_w
            bw = int(col_w - label_w - bar_gap)
            _draw_bar(panel, bx, yy - 2, bw, bar_h, float(au_probs[idx]), thr)

        # advance y to bottom of AU section
        used_rows = min(rows_per_col, len(INDEX_LIST))
        y = int(y + used_rows * row_h + 8)

    # Controls box at bottom
    # compute starting y so it sits above the bottom padding
    # controls = [
    #     "Controls",
    #     "SPACE  : Play/Pause",
    #     "← / →  : Step frame",
    #     "Drag Seek bar",
    #     "Q      : Quit"
    # ]
    # # Reserve ~ (len*22 + header extra)
    # needed = 24 + (len(controls)-1)*22 + 12
    # y_ctrl = max(y + 6, h - needed)
    # # box background
    # cv2.rectangle(panel, (pad-4, y_ctrl-22), (w-pad+4, h-pad), (30,30,30), -1)
    # cv2.rectangle(panel, (pad-4, y_ctrl-22), (w-pad+4, h-pad), (80,80,80), 1, cv2.LINE_AA)

    # # header
    # cv2.putText(panel, controls[0], (pad, y_ctrl), font, 0.7, (255,255,255), 2, cv2.LINE_AA)
    # yy = y_ctrl + 26
    # for line in controls[1:]:
    #     cv2.putText(panel, line, (pad, yy), font, 0.56, (220,220,220), 1, cv2.LINE_AA)
    #     yy += 22

    return panel

def draw_full_overlay(frame_bgr, au_probs, faces_norm_landmarks, *,
                      playback_fps, infer_fps, frame_idx, total_frames, run_time_s,
                      panel_w=520, au_cols=2):
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    # muscles (if landmarks available)
    if faces_norm_landmarks:
        active_names = compute_active_muscle_names(au_probs)
        if active_names:
            for face in faces_norm_landmarks:
                for mpair in ALL_MUSCLE_PAIRS:
                    if mpair.name not in active_names:
                        continue  # skip inactive regions
                    if len(mpair.upper) < 2 or len(mpair.lower) < 2:
                        continue
                    overlay = draw_muscle_patch(
                        overlay, face,
                        upper_indices=mpair.upper, lower_indices=mpair.lower,
                        color=mpair.color, core_alpha=mpair.core_alpha, edge_alpha=mpair.edge_alpha,
                        inner_width_px=int(mpair.inner_w), outer_width_px=int(mpair.outer_w),
                        samples_per_seg=mpair.samples, feather_extra_px=mpair.feather,
                        cut_upper_indices=mpair.upper_unfilled, cut_lower_indices=mpair.lower_unfilled,
                        cut_type=mpair.cut_type, cut_feather_px=mpair.cut_feather_px, cut_dilate_px=mpair.cut_dilate_px,
                        cut_use_catmull=mpair.cut_use_catmull, cut_samples_per_seg=mpair.cut_samples_per_seg
                    )
    # right sidebar (wider + columns)
    panel = make_info_panel(
        h, panel_w,
        au_probs=au_probs, playback_fps=playback_fps, infer_fps=infer_fps,
        frame_idx=frame_idx, total_frames=total_frames, run_time_s=run_time_s,
        cols=au_cols
    )

    # compose side-by-side
    out = np.concatenate([overlay, panel], axis=1)
    return out


def compute_active_muscle_names(au_probs) -> set[str]:
    """Return the set of muscle pair names to draw, based on AU thresholds."""
    active = set()
    for code, names in AU_TO_MUSCLES.items():
        idx = CODE_TO_IDX.get(code)
        if idx is None or idx >= len(au_probs):
            continue
        thr = THRESH_BY_CODE.get(code, 0.5)   # default fallback if not listed
        if float(au_probs[idx]) >= float(thr):
            active.update(names)
    return active
