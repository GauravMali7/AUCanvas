<div align="center">

  <!-- Title -->
  <h1>
    AUCanvas
  </h1>
  <h2>
    Facial Action Unit Detection & Visualization Framework
  </h2>

  <!-- Affiliation -->
  <p>
    Created by Awakening AI
  </p>

  <!-- Logo -->
  <!-- Update the logo path if needed -->
  <img src="docs/au-canvas-logo.png" alt="AUCanvas Logo" width="300"/>
  <br/><br/>

</div>


## 📸 Showcase
### Example running on a RTX 3090 GPU (Avg. FPS>50):
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
<!--   <tr> -->
      <td>
          <video src="https://github.com/user-attachments/assets/4cc05271-1758-4949-be40-fb6afe3274ed" width="100%" controls autoplay loop></video>
      </td>
<!--   </tr> -->
</table>

---

## 📢 News
- 🚀 Initial release of the AUCanvas viewer code and onnx model
- 📦 Upcoming: Qt interface  

---

## 📝 TODO
- ✅ Release the viewer code  
- ✅ Release the pretrained model  
- ☐ Release the Qt interface  

---

## 📋 Table of Contents
- [Installation](#installation)  
- [Running](#running)  
- [Customized Inference](#customized-inference)  
- [Citation](#citation)  
- [Acknowledgements](#acknowledgements)  

---

## 🛠️ Installation

**Requirements:**

* Python ≥ 3.10
* (Optional) CUDA 12.x + cuDNN 9.x for GPU inference

### 1. Create Environment

```bash
conda create -n au-canvas python=3.10
conda activate au-canvas
```

### 2. Install Dependencies

#### 📌 Option A: GPU (Linux/Windows with CUDA)

If you have a supported NVIDIA GPU and CUDA 12.x installed:

```bash
pip install mediapipe==0.10.14 \
    numpy==1.22.0 \
    onnx==1.17.0 \
    onnxruntime-gpu==1.22.0 \
    opencv-contrib-python==4.10.0.84 \
    opencv-python==4.10.0.84 \
    opencv-python-headless==4.10.0.84
```

#### 📌 Option B: CPU-only (Linux/macOS/Windows without CUDA)

For machines without GPU or on macOS:

```bash
pip install mediapipe==0.10.14 \
    numpy==1.22.0 \
    onnx==1.17.0 \
    onnxruntime==1.22.0 \
    opencv-contrib-python==4.10.0.84 \
    opencv-python==4.10.0.84 \
    opencv-python-headless==4.10.0.84
```

---

**Downloading Models:**  
Download the following checkpoints and put them inside the folder './model_weights'. 

- [FAU Detector Onnx](https://drive.google.com/file/d/1UIBcUm4EkgRz5OyZFL59HVaoxI3NNhl9/view?usp=sharing)

- [MediaPipe Facial Landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)


## 🎯 Running

Here’s a handy cheat-sheet of commands to run your program.

---

## Offline (video file)

**1) Display WITH panel (no saving)**

```bash
python3 video_viewer.py \
  --video /path/to/video.mp4 \
  --onnx_au model_weights/FAU.onnx \
  --mp_task model_weights/face_landmarker.task \
  --proc_h 1024 \
  --with_panel_display \
  --offline_pacing --offline_skip_policy drop
```

**) Save OVERLAY WITH panel, display WITHOUT panel**

```bash
python3 video_viewer.py \
  --video /path/to/video.mp4 \
  --onnx_au model_weights/FAU.onnx \
  --mp_task model_weights/face_landmarker.task \
  --proc_h 512 \
  --with_panel_save --save_with_overlay \
  --save_video ./out_overlay_panel.mp4 \
  --csv_out ./out_AUs.csv
```


---

## Online (live camera)

> Press **q** to quit the window. Replace `0` with your camera index if needed.

**1) Display WITH panel**

```bash
python3 video_viewer.py \
  --camera 0 \
  --onnx_au model_weights/FAU.onnx \
  --mp_task model_weights/face_landmarker.task \
  --proc_h 1024 \
  --with_panel_display \
  --save_video ./cam_raw.mp4 \
  --csv_out ./cam_raw_AUs.csv
```

**2) Fast path live: NO panel (MP off, light striding)**

```bash
python3 video_viewer.py \
  --camera 0 \
  --onnx_au model_weights/FAU.onnx \
  --disable_mp \
  --proc_h 1024 \
  --infer_stride 2 \
  --ui_stride 2 \
  --live_mp_stride 3 \
  --save_video ./cam_raw.mp4 \
  --csv_out ./cam_raw_AUs.csv
```

---

### Flag quick reference

* `--with_panel_display` / `--with_panel_save`: show/save the right info panel.
* `--save_with_overlay`: save whatever is drawn (panel and/or muscles). Omit to save raw frames.
* `--proc_h 512`: process & draw at height 512 (keeps aspect).
* `--infer_stride N`: run AU model every N frames (reuse last probs between).
* `--ui_stride N`: rebuild visualization every N frames.
* `--disable_mp` / `--live_mp_stride N`: turn off or stride MediaPipe landmarks.
* `--offline_pacing --offline_skip_policy drop`: smooth offline playback and drop backlog to keep up.
* `--fast_writer`: MJPG codec (faster writes, larger files).
* `--csv_out path.csv`: AU log file.
* `--save_video path`: output video file.
* `--start seconds`: start time for offline video.


---
## ⚡ Inference Speed
| Device           | Speed    |
| ---------------- | -------- |
| **RTX 3090 GPU** | > 50 FPS |
| **CPU only**     | > 5 FPS  |


---

## 🧪 Customized Inference

*(Coming soon)*
Guidelines for running **real-time** or **offline inference** with custom settings.

---

## 🖊️ Citation

If you find this work useful in your research, please cite:

```bibtex
@inproceedings{luo2022learning,
  title     = {Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition},
  author    = {Luo, Cheng and Song, Siyang and Xie, Weicheng and Shen, Linlin and Gunes, Hatice},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, {IJCAI-22}},
  pages     = {1239--1246},
  year      = {2022}
}
```

---

## 🤝 Acknowledgements

We gratefully acknowledge the following open-source projects:

* [ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU)
* [OpenGraphAU](https://github.com/lingjivoo/OpenGraphAU)
* [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)



