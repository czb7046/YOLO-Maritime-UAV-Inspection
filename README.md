
# Perception-Driven UAV Navigation for Maritime Vessel Inspection 🚁🚢

[![Paper](https://img.shields.io/badge/Paper-Ocean_Engineering-blue.svg)](URL_TO_YOUR_PAPER)
[![Dataset](https://img.shields.io/badge/Dataset-Public_Available-green.svg)](URL_TO_YOUR_DATASET)
[![Platform](https://img.shields.io/badge/Hardware-Jetson_Orin_Nano_Super-orange.svg)](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit)

Official repository for the paper: **"Guided Search and Edge YOLO Benchmarking for UAV-Based Maritime Vessel Inspection: A Perception-Driven Navigation Strategy"** (Under Review in *Ocean Engineering*).

## 🌟 Highlights
- **The "Data Augmentation Trap":** Discovered and resolved the systemic "Lateral (Port-Starboard) Confusion" in maritime vision by eliminating physics-violating horizontal flip augmentations.
- **Edge Benchmarking (Thermal Equilibrium):** Evaluated 50 YOLO architectures (v8-v12) on NVIDIA Jetson Orin Nano Super under strict 60-second sustained loads.
- **Adaptive WNS Protocol:** Proposed a Weighted Normalized Scoring (WNS) system, identifying **`YOLOv9t-engine`** as the Pareto-optimal model (0.879 mAP@50, 72.78 FPS, 5.98 W).
- **Sequence-Level Simulation:** Quantitatively proved that our Perception-Initialized Guided Search Strategy (Initial-GSS) reduces the UAV target search time by **44.7%**.

---

## 📂 Repository Structure
- `scripts/`: Contains python scripts for training, TensorRT exporting, edge benchmarking, and Monte Carlo simulation.
- `logs/`: Contains the raw `tegrastats` power logs, ensuring 100% statistical reproducibility for the 60s thermal equilibrium tests.
- `models/`: Pre-trained weights for the Pareto-optimal models.
- `dataset/`: Instructions and links to download the complete UAV-based Vessel Perspectives Dataset.

---

## 🛠️ Hardware & Software Requirements

To perfectly reproduce the edge evaluation environments, the following setup is required:
- **Hardware:** NVIDIA Jetson Orin Nano Super (8GB Memory).
- **OS:** Native Ubuntu 22.04.4 LTS (JetPack 6.x).
- **Docker:** `ultralytics/ultralytics:latest-jetson-jetpack6` (Docker engine version < 28.0).

---

## 🚀 Quick Start / Reproducibility Guide

### 1. Dataset Access
The newly constructed UAV-based Vessel Perspectives Dataset (1,967 annotated images, 5 classes: Stern, Bow, Left, Right, Top) is publicly available.
🔗 **[Download Dataset Here](https://drive.google.com/file/d/12KhCTpByShqz4BqTaFJqe7rzCMvGBwgN/view)**

### 2. Training with Domain-Specific Adaptation
To avoid the "Lateral Confusion" trap, horizontal flipping must be disabled.
```bash
# In your training script or CLI:
yolo task=detect mode=train model=yolov9t.pt data=dataset.yaml epochs=80 imgsz=320 fliplr=0.0
```

### 3. Edge TensorRT Export
We enforce FP16 precision and a fixed workspace to guarantee edge determinism.
```python
from ultralytics import YOLO
model = YOLO('yolov9t.pt')
model.export(format='engine', imgsz=320, half=True, workspace=4, dynamic=False)
```

### 4. 60-Second Thermal Equilibrium Benchmark
Run the automated benchmarking script inside the Jetson Docker container. This script continuously feeds a video stream to the GPU and records `VDD_IN` power via `tegrastats`.
```bash
python scripts/auto_benchmark.py
```

### 5. WNS Scoring & Sequence-Level Simulation
Calculate the WNS ranking and run the timeline-based Monte Carlo simulation (comparing BSS vs. Initial-GSS) to validate the endurance gains.
```bash
python scripts/verify_wns.py
python scripts/sequence_simulation.py
```

---

## 📜 Citation
If you find this code or dataset useful for your research, please cite our paper:
```bibtex
@article{cao2024guided,
  title={Guided Search and Edge YOLO Benchmarking for UAV-Based Maritime Vessel Inspection: A Perception-Driven Navigation Strategy},
  author={Cao, Zhi-Bo},
  journal={Ocean Engineering},
  year={2024},
  note={Under Review}
}
```




