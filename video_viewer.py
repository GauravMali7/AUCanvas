# viewer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, argparse, threading, queue, cv2, numpy as np
import csv
from dataclasses import dataclass
from src.au_detector import AUDetector
from src.draw_utils import draw_full_overlay

# Optional MediaPipe
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _HAS_MP = True
except Exception:
    _HAS_MP = False

@dataclass
class FrameItem:
    idx: int
    frame: np.ndarray
    au_probs: np.ndarray
    infer_fps: float
    faces_norm_landmarks: list | None
    video_time_s: float

class FrameWorker:
    def __init__(self, cap: cv2.VideoCapture, detector: AUDetector, fps: float, q: queue.Queue,
                 mp_task_path: str | None = None):
        self.cap = cap
        self.detector = detector
        self.fps = max(1e-6, fps)
        self.q = q
        self.stop_event = threading.Event()
        self.seek_lock = threading.Lock()
        self.seek_to: int | None = None
        self.eof_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._mp_face = None
        if mp_task_path and _HAS_MP and os.path.isfile(mp_task_path):
            base_opts = mp_python.BaseOptions(model_asset_path=mp_task_path)
            face_opts = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts, num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                running_mode=mp_vision.RunningMode.VIDEO
            )
            self._mp_face = mp_vision.FaceLandmarker.create_from_options(face_opts)
            print(f"[INFO] MediaPipe FaceLandmarker loaded: {mp_task_path}")
        else:
            if mp_task_path and not os.path.isfile(mp_task_path):
                print(f"[WARN] MediaPipe .task not found; landmark overlays disabled.")
            if not _HAS_MP and mp_task_path:
                print(f"[WARN] mediapipe not installed; landmark overlays disabled.")

    def start(self): self.thread.start()
    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)
    def request_seek(self, frame_idx: int):
        with self.seek_lock:
            self.seek_to = max(0, int(frame_idx))
        try:
            while True: self.q.get_nowait()
        except queue.Empty:
            pass
    def _apply_seek_if_any(self):
        with self.seek_lock:
            target = self.seek_to; self.seek_to = None
        if target is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.eof_event.clear()

    def _run(self):
        while not self.stop_event.is_set():
            self._apply_seek_if_any()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.eof_event.set()
                time.sleep(0.003)
                continue
            cur_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            t0 = time.perf_counter()
            au_probs = self.detector.infer(frame)
            infer_fps = 1.0 / max(1e-6, (time.perf_counter() - t0))
            faces_norm = None
            if self._mp_face is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int(round(cur_frame / self.fps * 1000.0))
                try:
                    result = self._mp_face.detect_for_video(mp_img, ts_ms)
                    if result and result.face_landmarks:
                        faces_norm = [[(lm.x, lm.y, lm.z) for lm in fl] for fl in result.face_landmarks]
                except Exception:
                    faces_norm = None
            item = FrameItem(
                idx=cur_frame, frame=frame, au_probs=au_probs,
                infer_fps=infer_fps, faces_norm_landmarks=faces_norm,
                video_time_s=cur_frame / self.fps
            )
            while not self.stop_event.is_set():
                try:
                    self.q.put_nowait(item); break
                except queue.Full:
                    try: self.q.get_nowait()
                    except queue.Empty: pass

def main():
    ap = argparse.ArgumentParser(description="AU Viewer with wide info panel")
    ap.add_argument("--video", required=True)
    ap.add_argument("--onnx_au", default="mental_health/FAU.onnx")
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--start", type=float, default=0.0, help="Start time (s)")
    ap.add_argument("--buffer_sec", type=float, default=30.0)
    ap.add_argument("--mp_task", default="", help="MediaPipe FaceLandmarker .task (optional)")
    # NEW: panel width and AU columns
    ap.add_argument("--panel_w", type=int, default=700, help="Right panel width in pixels")
    ap.add_argument("--au_cols", type=int, default=1, help="Number of AU columns on the panel")
    ap.add_argument("--csv_out", default="", help="CSV output path (default: <video>_AUs_log.csv)")  # ← ADD

    args = ap.parse_args()

    if not os.path.isfile(args.video): raise FileNotFoundError(args.video)
    if not os.path.isfile(args.onnx_au): raise FileNotFoundError(args.onnx_au)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError("Failed to open video")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # seek to start (seconds)
    if args.start > 0:
        start_frame = max(0, min(total_frames-1, int(round(args.start * fps))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        start_frame = 0

    detector = AUDetector(args.onnx_au, force_cpu=args.force_cpu)

    buffer_frames = max(1, int(args.buffer_sec * fps))
    q: queue.Queue[FrameItem] = queue.Queue(maxsize=buffer_frames)
    worker = FrameWorker(cap, detector, fps, q, mp_task_path=(args.mp_task or None))
    worker.start()

    win = "AU Video Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, max(1024, width + args.panel_w), max(576, height))

    playing = True
    cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    suppress_seek = False

    # Accumulated run-time counter
    run_time_s = 0.0
    last_tick = time.perf_counter()

    # Playback FPS (consumer)
    fps_win = 0.0
    fps_alpha = 0.15
    last_vis = np.zeros((height, width + args.panel_w, 3), dtype=np.uint8)

    def on_seek(pos):
        nonlocal playing, suppress_seek, cur_frame, last_tick
        if suppress_seek: return
        playing = False
        worker.request_seek(pos)
        cur_frame = pos
        last_tick = time.perf_counter()

    cv2.createTrackbar("Seek", win, start_frame, max(0, total_frames - 1), on_seek)

    print("[INFO] Controls: SPACE=play/pause | ←/→ step | Drag seek | q quit")

    from src.au_config import INDEX_LIST, THRESHOLDS, AU_INDEX

    # Map index -> threshold and (index, code) pairs for header
    thr_map = {idx: thr for idx, thr in zip(INDEX_LIST, THRESHOLDS)}
    au_meta = [(idx, AU_INDEX[idx] if idx < len(AU_INDEX) else str(idx)) for idx in INDEX_LIST]

    # Buffer rows keyed by frame index to avoid duplicates on seeks
    rows_by_frame: dict[int, list] = {}  # frame_idx -> row list

    try:
        while True:
            now = time.perf_counter()
            if playing:
                dt = now - last_tick
                run_time_s += dt
                last_tick = now
            else:
                last_tick = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                playing = not playing
            elif key == ord('q'):
                break
            elif key == 81:  # LEFT
                playing = False
                target = max(0, cur_frame - 1)
                worker.request_seek(target); cur_frame = target
            elif key == 83:  # RIGHT
                playing = False
                target = min(total_frames - 1, cur_frame + 1)
                worker.request_seek(target); cur_frame = target

            got = False
            if playing:
                try:
                    item: FrameItem = q.get_nowait()
                    got = True
                except queue.Empty:
                    pass

            if got:
                cur_frame = item.idx
                # simple playback FPS (smoothed)
                inst = 1.0 / max(1e-6, (time.perf_counter() - now))
                fps_win = inst * fps_alpha + fps_win * (1.0 - fps_alpha)

                vis = draw_full_overlay(
                    item.frame, item.au_probs, item.faces_norm_landmarks,
                    playback_fps=fps_win, infer_fps=item.infer_fps,
                    frame_idx=item.idx, total_frames=total_frames,
                    run_time_s=run_time_s,
                    panel_w=args.panel_w, au_cols=args.au_cols
                )
                last_vis = vis

                suppress_seek = True
                cv2.setTrackbarPos("Seek", win, max(0, min(total_frames - 1, cur_frame)))
                suppress_seek = False

                row = [cur_frame]
                for idx, code in au_meta:
                    prob = float(item.au_probs[idx]) if idx < len(item.au_probs) else 0.0
                    thr  = float(thr_map.get(idx, 0.5))  # default to 0.5 if not listed
                    act  = 1 if prob >= thr else 0
                    row.extend([f"{prob:.5f}", act])
                rows_by_frame[cur_frame] = row

            cv2.imshow(win, last_vis)

            if worker.eof_event.is_set() and q.empty() and playing:
                print("[INFO] End of video.")
                break

    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()

                # ----- Save CSV on exit -----
        try:
            base = os.path.splitext(os.path.basename(args.video))[0]
            if args.csv_out and (len(args.csv_out) > 0):
                csv_path = args.csv_out or f"{base}_AUs_log.csv"
            else:
                video_name = args.video.split('/')[-1][:-4]
                csv_path = f"{video_name}_AUs_log.csv"

            # header: frame_idx, AU1_prob, AU1_act, AU2_prob, AU2_act, ...
            header = ["frame_idx"]
            for _, code in au_meta:
                header += [f"AU{code}_prob", f"AU{code}_act"]

            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                for fi in sorted(rows_by_frame.keys()):
                    w.writerow(rows_by_frame[fi])

            print(f"[INFO] Saved CSV: {csv_path}  (rows={len(rows_by_frame)})")
        except Exception as e:
            print(f"[WARN] Failed to save CSV: {e}")
        # --------------------------------


if __name__ == "__main__":
    main()
