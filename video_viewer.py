# Copyright (c) 2025 Awakening AI

# video_viewer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, argparse, threading, queue, cv2, numpy as np
import csv
from dataclasses import dataclass, field
from types import SimpleNamespace
from au_canvas.au_detector import AUDetector
from au_canvas.draw_utils import draw_full_overlay, compute_active_muscle_names, draw_muscle_patch
from au_canvas.paint_config import ALL_MUSCLE_PAIRS

# Optional MediaPipe
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _HAS_MP = True
except Exception:
    _HAS_MP = False


# ---------- Helpers ----------
def _resize_to_h(img: np.ndarray, target_h: int) -> np.ndarray:
    """Resize image to target height (keep aspect)."""
    if img is None or target_h is None or target_h <= 0:
        return img
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = int(round(w * (target_h / float(h))))
    interp = cv2.INTER_AREA if target_h < h else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, target_h), interpolation=interp)


# ---------- Shared HUD ----------
@dataclass
class HUDState:
    cur_frame: int = 0
    total_frames: int = 0
    run_time_s: float = 0.0
    playback_fps: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> SimpleNamespace:
        with self._lock:
            return SimpleNamespace(
                cur_frame=self.cur_frame,
                total_frames=self.total_frames,
                run_time_s=self.run_time_s,
                playback_fps=self.playback_fps
            )


# ---------- Data Types ----------
@dataclass
class FrameItem:
    idx: int
    frame: np.ndarray
    au_probs: np.ndarray
    infer_fps: float
    faces_norm_landmarks: list | None
    video_time_s: float


@dataclass
class RenderItem:
    idx: int
    display_frame: np.ndarray
    save_frame: np.ndarray
    au_probs: np.ndarray
    infer_fps: float
    faces_norm_landmarks: list | None


# ---------- Worker: Capture + Inference ----------
class FrameWorker:
    def __init__(self,
                 cap: cv2.VideoCapture,
                 detector: AUDetector,
                 fps: float,
                 q: queue.Queue,
                 mp_task_path: str | None = None,
                 live_mode: bool = False,
                 proc_h: int = 0,
                 live_mp_stride: int = 1,
                 offline_stride: int = 1,
                 infer_stride: int = 1):
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
        self.live_mode = live_mode

        self._live_idx = 0
        self._proc_h = int(proc_h or 0)
        self._live_mp_stride = max(1, int(live_mp_stride))
        self._offline_stride = max(1, int(offline_stride))
        self._infer_stride = max(1, int(infer_stride))

        self._last_au = None
        self._last_faces = None

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
        if self.live_mode:
            return  # seeking not supported in live mode
        with self.seek_lock:
            self.seek_to = max(0, int(frame_idx))
        try:
            while True: self.q.get_nowait()
        except queue.Empty:
            pass

    def _apply_seek_if_any(self):
        if self.live_mode:
            return
        with self.seek_lock:
            target = self.seek_to; self.seek_to = None
        if target is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            self.eof_event.clear()
            # reset caches after seek
            self._last_au = None
            self._last_faces = None

    def _run(self):
        while not self.stop_event.is_set():
            self._apply_seek_if_any()
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if not self.live_mode:
                    self.eof_event.set()
                time.sleep(0.003)
                continue

            if self._proc_h > 0:
                frame = _resize_to_h(frame, self._proc_h)

            if self.live_mode:
                cur_frame = self._live_idx
                self._live_idx += 1
            else:
                cur_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            if (not self.live_mode) and (self._offline_stride > 1):
                if (cur_frame % self._offline_stride) != 0:
                    continue

            # AU inference (stride/cache)
            do_infer = (self._last_au is None) or (cur_frame % self._infer_stride == 0)
            t0 = time.perf_counter()
            if do_infer:
                au_probs = self.detector.infer(frame)
                self._last_au = au_probs
            else:
                au_probs = self._last_au
            infer_fps = 1.0 / max(1e-6, (time.perf_counter() - t0)) if do_infer else float('inf')

            # MediaPipe (stride/cache)
            faces_norm = None
            if self._mp_face is not None:
                should_run_mp = (not self.live_mode) or (cur_frame % self._live_mp_stride == 0) or (self._last_faces is None)
                if should_run_mp:
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        ts_ms = int(round(cur_frame / self.fps * 1000.0))
                        result = self._mp_face.detect_for_video(mp_img, ts_ms)
                        if result and result.face_landmarks:
                            faces_norm = [[(lm.x, lm.y, lm.z) for lm in fl] for fl in result.face_landmarks]
                        self._last_faces = faces_norm
                    except Exception:
                        faces_norm = self._last_faces
                else:
                    faces_norm = self._last_faces

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


# ---------- Worker: Rendering ----------
class RenderWorker:
    """Consumes FrameItem(s), produces precomposed display/save frames."""
    def __init__(self,
                 in_q: queue.Queue,
                 out_q: queue.Queue,
                 stop_event: threading.Event,
                 hud: HUDState,
                 *,
                 panel_w: int,
                 au_cols: int,
                 ui_stride: int,
                 with_panel_display: bool,
                 with_panel_save: bool,
                 save_with_overlay: bool,
                 no_muscles: bool):
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.hud = hud

        self.panel_w = int(panel_w)
        self.au_cols = int(au_cols)
        self.ui_stride = max(1, int(ui_stride))
        self.with_panel_display = bool(with_panel_display)
        self.with_panel_save = bool(with_panel_save)
        self.save_with_overlay = bool(save_with_overlay)
        self.no_muscles = bool(no_muscles)

        self.thread = threading.Thread(target=self._run, daemon=True)
        self._last_display = None
        self._last_save = None
        self._vis_counter = 0

    def start(self): self.thread.start()
    def join(self, timeout=None): self.thread.join(timeout=timeout)

    def _compose_no_panel(self, frame_bgr, au_probs, faces_norm):
        """Fast path: no panel; optionally draw muscles (lightweight vs sidebar)."""
        if self.no_muscles or not faces_norm:
            return frame_bgr
        overlay = frame_bgr.copy()
        try:
            active_names = compute_active_muscle_names(au_probs)
        except Exception:
            return frame_bgr
        if not active_names:
            return frame_bgr
        for face in faces_norm:
            for mpair in ALL_MUSCLE_PAIRS:
                if mpair.name not in active_names:
                    continue
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
        return overlay

    def _run(self):
        while not self.stop_event.is_set():
            try:
                item: FrameItem = self.in_q.get(timeout=0.02)
            except queue.Empty:
                continue

            self._vis_counter += 1
            rebuild_vis = (self._vis_counter % self.ui_stride == 0) or (self._last_display is None) or (self._last_save is None)

            display_frame = self._last_display
            save_frame = self._last_save

            if rebuild_vis:
                need_panel_any = self.with_panel_display or (self.with_panel_save and self.save_with_overlay)

                if need_panel_any:
                    hud = self.hud.snapshot()
                    # Use the current item's frame index; use HUD totals & timing so panel numbers are correct
                    full = draw_full_overlay(
                        item.frame, item.au_probs, (None if self.no_muscles else item.faces_norm_landmarks),
                        playback_fps=hud.playback_fps, infer_fps=item.infer_fps,
                        frame_idx=item.idx, total_frames=hud.total_frames, run_time_s=hud.run_time_s,
                        panel_w=self.panel_w, au_cols=self.au_cols
                    )
                    no_panel = full[:, :item.frame.shape[1]]
                    display_frame = full if self.with_panel_display else no_panel
                    if self.save_with_overlay:
                        save_frame = full if self.with_panel_save else no_panel
                    else:
                        save_frame = item.frame
                else:
                    base = self._compose_no_panel(item.frame, item.au_probs, item.faces_norm_landmarks)
                    display_frame = base
                    save_frame = (base if self.save_with_overlay else item.frame)

                self._last_display = display_frame
                self._last_save = save_frame

            out = RenderItem(
                idx=item.idx,
                display_frame=display_frame,
                save_frame=save_frame,
                au_probs=item.au_probs,
                infer_fps=item.infer_fps,
                faces_norm_landmarks=item.faces_norm_landmarks
            )
            try:
                self.out_q.put_nowait(out)
            except queue.Full:
                try: _ = self.out_q.get_nowait()
                except queue.Empty: pass
                try: self.out_q.put_nowait(out)
                except queue.Full: pass


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="AU Viewer (multithreaded render + accurate panel HUD)")
    # Source selection
    ap.add_argument("--video", help="Path to offline video file (use this OR --camera)")
    ap.add_argument("--camera", type=int, default=-1, help="Camera index for LIVE mode (>=0).")
    # Performance knobs
    ap.add_argument("--proc_h", type=int, default=512, help="Resize ALL frames (live+offline) to this height for processing & drawing.")
    ap.add_argument("--infer_stride", type=int, default=1, help="Run AU inference every N frames (reuse last probs otherwise).")
    ap.add_argument("--ui_stride", type=int, default=1, help="Rebuild visualization every N frames (reuse last frame otherwise).")
    ap.add_argument("--no_muscles", action="store_true", help="Disable muscle patches overlay (much faster).")
    ap.add_argument("--cv_threads", type=int, default=0, help="Set OpenCV thread count (0=leave default).")
    # Model & runtime
    ap.add_argument("--onnx_au", default="model_weights/FAU.onnx")
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--start", type=float, default=0.0, help="Start time (s) for offline video")
    ap.add_argument("--buffer_sec", type=float, default=30.0, help="Offline buffering seconds")
    ap.add_argument("--mp_task", default="model_weights/face_landmarker.task", help="MediaPipe FaceLandmarker .task (optional)")
    ap.add_argument("--disable_mp", action="store_true", help="Disable MediaPipe (faster).")
    ap.add_argument("--live_mp_stride", type=int, default=2, help="Run MediaPipe every N frames in LIVE mode (>=1).")
    # Offline smoothness
    ap.add_argument("--offline_pacing", action="store_true", help="Pace offline playback at source FPS for smooth display.")
    ap.add_argument("--offline_skip_policy", choices=["none", "drop"], default="drop",
                    help="When behind schedule in offline pacing, drop queued frames to catch up.")
    ap.add_argument("--offline_stride", type=int, default=1, help="Process every Nth frame in offline (>=1).")
    # Panel layout + per-output panel control
    ap.add_argument("--panel_w", type=int, default=560, help="Right panel width baseline.")
    ap.add_argument("--au_cols", type=int, default=1, help="Number of AU columns on the panel.")
    ap.add_argument("--with_panel_display", action="store_true", help="Display with side panel.")
    ap.add_argument("--with_panel_save", action="store_true", help="Save with side panel (when saving overlay).")
    # CSV log
    ap.add_argument("--csv_out", default="", help="CSV output path (default: <source>_AUs_log.csv)")
    # Saving video
    ap.add_argument("--save_video", default="", help="Optional path to write output video (e.g., out.mp4).")
    ap.add_argument("--save_with_overlay", action="store_true", help="If set, save the DRAWN/visualized frames. Otherwise, save raw frames.")
    ap.add_argument("--fast_writer", action="store_true", help="Use MJPG for faster disk writes (bigger files).")

    args = ap.parse_args()

    # OpenCV tuning
    cv2.setUseOptimized(True)
    if args.cv_threads and args.cv_threads > 0:
        try: cv2.setNumThreads(int(args.cv_threads))
        except Exception: pass

    # Validate source
    using_camera = args.camera >= 0
    using_video = bool(args.video)
    if using_camera == using_video:
        raise ValueError("Specify exactly one source: either --video <file> OR --camera <index>.")

    if using_video and not os.path.isfile(args.video):
        raise FileNotFoundError(args.video)
    if not os.path.isfile(args.onnx_au):
        raise FileNotFoundError(args.onnx_au)

    # Open source
    if using_camera:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera index {args.camera}")

        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception: pass

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3 or np.isnan(fps):
            fps = 30.0  # fallback

        src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
        src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        height = int(args.proc_h) if args.proc_h and args.proc_h > 0 else src_h
        width  = int(round(src_w * (height / max(1.0, float(src_h)))))
        total_frames = 0
        start_frame = 0
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        height = int(args.proc_h) if args.proc_h and args.proc_h > 0 else sh
        width  = int(round(sw * (height / max(1.0, float(sh)))))

        if args.start > 0:
            start_frame = max(0, min(max(0, total_frames - 1), int(round(args.start * fps))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        else:
            start_frame = 0

    # Resolve MP enablement
    mp_task_path = None if (args.disable_mp or (args.mp_task == "")) else args.mp_task

    detector = AUDetector(args.onnx_au, force_cpu=args.force_cpu)

    # Queues: tiny for live, deeper for offline
    if using_camera:
        buffer_frames = 2
    else:
        buffer_frames = max(1, int(min(120, args.buffer_sec * fps)))

    infer_q: queue.Queue[FrameItem] = queue.Queue(maxsize=buffer_frames)
    worker = FrameWorker(
        cap, detector, fps, infer_q,
        mp_task_path=(mp_task_path or None),
        live_mode=using_camera,
        proc_h=args.proc_h,
        live_mp_stride=args.live_mp_stride,
        offline_stride=args.offline_stride,
        infer_stride=args.infer_stride
    )
    worker.start()

    # HUD shared state
    hud = HUDState()
    hud.update(total_frames=(total_frames if not using_camera else 0))

    render_q: queue.Queue[RenderItem] = queue.Queue(maxsize=8)
    stop_event = threading.Event()
    renderer = RenderWorker(
        infer_q, render_q, stop_event, hud,
        panel_w=args.panel_w, au_cols=args.au_cols, ui_stride=args.ui_stride,
        with_panel_display=args.with_panel_display,
        with_panel_save=args.with_panel_save,
        save_with_overlay=args.save_with_overlay,
        no_muscles=args.no_muscles
    )
    renderer.start()

    win = "AU Viewer (LIVE)" if using_camera else "AU Video Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, max(960, width + (args.with_panel_display and args.save_with_overlay and args.with_panel_display) * args.panel_w), max(540, height))

    playing = True
    cur_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) if using_video else 0
    suppress_seek = False

    # UI timers / pacing
    run_time_s = 0.0
    last_tick = time.perf_counter()
    # Playback FPS (based on display cadence)
    disp_fps = 0.0
    disp_alpha = 0.2
    last_display_t = time.perf_counter()

    last_display = np.zeros((height, width, 3), dtype=np.uint8)

    frame_interval = (1.0 / fps) if fps > 1e-6 else 1.0/25.0
    next_show_time = time.perf_counter() + frame_interval

    # Seek (offline)
    def on_seek(pos):
        nonlocal playing, suppress_seek, cur_frame, last_tick, next_show_time, run_time_s
        if suppress_seek or using_camera: return
        playing = False
        worker.request_seek(pos)
        cur_frame = pos
        last_tick = time.perf_counter()
        next_show_time = last_tick + frame_interval
        # HUD updates on next display

    if not using_camera:
        cv2.createTrackbar("Seek", win, start_frame, max(0, total_frames - 1), on_seek)

    print("[INFO] Controls: SPACE=play/pause | ←/→ step (offline) | q quit")

    from au_canvas.au_config import INDEX_LIST, THRESHOLDS, AU_INDEX
    thr_map = {idx: thr for idx, thr in zip(INDEX_LIST, THRESHOLDS)}
    au_meta = [(idx, AU_INDEX[idx] if idx < len(AU_INDEX) else str(idx)) for idx in INDEX_LIST]

    rows_by_frame: dict[int, list] = {}
    writer = None

    def _maybe_init_writer(frame_for_size: np.ndarray):
        nonlocal writer
        if writer is not None or not args.save_video:
            return
        h, w = frame_for_size.shape[:2]
        if args.fast_writer:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        else:
            ext = os.path.splitext(args.save_video)[1].lower()
            fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext in (".mp4", ".m4v", ".mov", "") else "XVID"))
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open writer for {args.save_video}")
        print(f"[INFO] Writing video to: {args.save_video} @ {fps:.2f} FPS, size=({w}x{h}), overlay={args.save_with_overlay}, panel_save={args.with_panel_save}, codec={'MJPG' if args.fast_writer else 'auto'}")

    try:
        while True:
            now = time.perf_counter()
            if playing:
                dt = now - last_tick
                run_time_s += dt
                last_tick = now
            else:
                last_tick = now

            # Update HUD runtime continuously
            hud.update(run_time_s=run_time_s)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                playing = not playing
                if playing and (not using_camera) and args.offline_pacing:
                    next_show_time = time.perf_counter() + frame_interval
            elif key == ord('q'):
                break
            elif not using_camera and key == 81:  # LEFT
                playing = False
                target = max(0, cur_frame - 1)
                worker.request_seek(target); cur_frame = target
            elif not using_camera and key == 83:  # RIGHT
                playing = False
                target = min(total_frames - 1, cur_frame + 1)
                worker.request_seek(target); cur_frame = target

            if (not using_camera) and args.offline_pacing and playing:
                ahead = next_show_time - time.perf_counter()
                if ahead > 0.001:
                    time.sleep(min(ahead, 0.01))

            got = False
            ritem: RenderItem | None = None
            if playing:
                try:
                    ritem = render_q.get_nowait()
                    while True:
                        try:
                            ritem = render_q.get_nowait()
                        except queue.Empty:
                            break
                    got = True
                except queue.Empty:
                    got = False

            if got and ritem is not None:
                cur_frame = ritem.idx

                # Display FPS based on real display cadence
                tnow = time.perf_counter()
                dt_disp = tnow - last_display_t
                last_display_t = tnow
                if dt_disp > 1e-6:
                    inst = 1.0 / dt_disp
                    disp_fps = inst * disp_alpha + disp_fps * (1.0 - disp_alpha)

                # Update HUD before next render compositions
                hud.update(cur_frame=cur_frame, playback_fps=disp_fps)

                last_display = ritem.display_frame

                if not using_camera:
                    suppress_seek = True
                    try:
                        cv2.setTrackbarPos("Seek", win, max(0, min(max(0, total_frames - 1), cur_frame)))
                    except Exception:
                        pass
                    suppress_seek = False

                # CSV row
                row = [cur_frame]
                for idx, code in au_meta:
                    prob = float(ritem.au_probs[idx]) if idx < len(ritem.au_probs) else 0.0
                    thr  = float(thr_map.get(idx, 0.5))
                    act  = 1 if prob >= thr else 0
                    row.extend([f"{prob:.5f}", act])
                rows_by_frame[cur_frame] = row

                # Saving
                _maybe_init_writer(ritem.save_frame)
                if writer is not None:
                    writer.write(ritem.save_frame)

                if (not using_camera) and args.offline_pacing:
                    now2 = time.perf_counter()
                    if now2 - next_show_time > 0.5:
                        next_show_time = now2 + frame_interval
                        if args.offline_skip_policy == "drop":
                            while render_q.qsize() > 1:
                                try: _ = render_q.get_nowait()
                                except queue.Empty: break
                    else:
                        next_show_time += frame_interval

            cv2.imshow(win, last_display)

            if (not using_camera) and worker.eof_event.is_set() and infer_q.empty() and render_q.empty() and playing:
                print("[INFO] End of video.")
                break

    finally:
        stop_event.set()
        worker.stop()
        renderer.join(timeout=1.0)
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        # ----- Save CSV on exit -----
        try:
            if using_camera:
                base_name = f"camera{args.camera}"
            else:
                base_name = os.path.splitext(os.path.basename(args.video))[0]

            csv_path = (args.csv_out if args.csv_out else f"{base_name}_AUs_log.csv")
            header = ["frame_idx"]
            from au_canvas.au_config import AU_INDEX, INDEX_LIST
            for idx in INDEX_LIST:
                code = AU_INDEX[idx] if idx < len(AU_INDEX) else str(idx)
                header += [f"AU{code}_prob", f"AU{code}_act"]

            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                for fi in sorted(rows_by_frame.keys()):
                    w.writerow(rows_by_frame[fi])

            print(f"[INFO] Saved CSV: {csv_path}  (rows={len(rows_by_frame)})")
        except Exception as e:
            print(f"[WARN] Failed to save CSV: {e}")


if __name__ == "__main__":
    main()
