# au_detector.py
import os
import onnx
import onnxruntime as ort
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class ORTModel:
    session: ort.InferenceSession
    input_name: str
    layout: str
    H: int
    W: int

    @staticmethod
    def create(path: str, force_cpu: bool=False) -> "ORTModel":
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        providers = ['CPUExecutionProvider'] if force_cpu else (
            ['CUDAExecutionProvider','CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        )
        sess = ort.InferenceSession(path, providers=providers)
        inp = sess.get_inputs()[0]
        in_shape = list(inp.shape)
        if len(in_shape) != 4:
            raise RuntimeError(f"Unsupported input rank: {in_shape}")
        if in_shape[1] == 3:
            layout, H, W = "NCHW", int(in_shape[2] or 224), int(in_shape[3] or 224)
        elif in_shape[3] == 3:
            layout, H, W = "NHWC", int(in_shape[1] or 224), int(in_shape[2] or 224)
        else:
            layout, H, W = "NCHW", 224, 224
        print(f"[INFO] Input '{inp.name}' expects {layout} {H}x{W}; Providers={sess.get_providers()}")
        return ORTModel(sess, inp.name, layout, H, W)

class AUDetector:
    def __init__(self, model_path: str, force_cpu: bool=False):
        self.model = ORTModel.create(model_path, force_cpu=force_cpu)

    def _preprocess(self, frame_bgr):
        H, W = self.model.H, self.model.W
        h, w = frame_bgr.shape[:2]
        scale = max(H / h, W / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        y0 = (nh - H) // 2
        x0 = (nw - W) // 2
        crop = resized[y0:y0+H, x0:x0+W]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        if self.model.layout == "NCHW":
            return np.transpose(rgb, (2,0,1))[None,...].astype(np.float32)
        else:
            return rgb[None,...].astype(np.float32)

    def infer(self, frame_bgr):
        inp = self._preprocess(frame_bgr)
        outputs = self.model.session.run(None, {self.model.input_name: inp})
        au = outputs[0][0].astype(np.float32)
        return np.clip(au, 0.0, 1.0)
