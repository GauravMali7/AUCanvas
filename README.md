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
- Python ≥ 3.10  
- CUDA (for GPU inference)  

```bash
# Create environment
conda create -n au-canvas python=3.10
conda activate au-canvas

# Install dependencies
pip install -r requirements.txt
````

---

**Downloading Models:**  
Download the following checkpoints and put them inside the folder './model_weights'. 

- [FAU Detector Onnx](https://drive.google.com/file/d/1UIBcUm4EkgRz5OyZFL59HVaoxI3NNhl9/view?usp=sharing)

- [MediaPipe Facial Landmarker](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)


## 🎯 Running

Example command for runing a video version FAU detection and visualization:

```bash
python video_viewer.py \
    --video example/demo.mp4 \
    --onnx_au model_weights/FAU.onnx \
    --mp_task model_weights/face_landmarker.task
```

Alternatively, use the shell script:

```bash
sh run_video_viewer.sh
```

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



