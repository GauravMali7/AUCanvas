<div align="center">

  <!-- Title -->
  <h1>
    AUCanvas
  </h1>
  <h2>
    Facial Action Unit Detection & Visualization Framework
  </h2>
  <br/>

  <!-- Affiliation -->
  <p>
    Created by Awakening AI
  </p>
  <br/>

  <!-- Logo -->
  <!-- Update the logo path if needed -->
  <img src="docs/aucanvas-logo.png" alt="AUCanvas Logo" width="300"/>
  <br/><br/>

</div>

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



