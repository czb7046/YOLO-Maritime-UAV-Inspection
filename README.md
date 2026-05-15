# Perception-Driven UAV Navigation for Maritime Vessel Inspection 🚁🚢

[![Paper](https://img.shields.io/badge/Paper-Ocean_Engineering-blue.svg)](URL_TO_YOUR_PAPER)
[![Dataset](https://img.shields.io/badge/Dataset-Public_Available-green.svg)](https://drive.google.com/file/d/1rm11-l1lWx8frvlJWpTZ8psUeAqdbzrG/view?usp=sharing)
[![Weights](https://img.shields.io/badge/Weights-PT_%26_Engine-orange.svg)](https://drive.google.com/file/d/1FEBnxlvqQmmZ7aioJIL1_O2xZLTWDVmc/view?usp=sharing)
[![Platform](https://img.shields.io/badge/Hardware-Jetson_Orin_Nano_Super-blue.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)

Official repository for the paper: **"Guided Search and Edge YOLO Benchmarking for UAV-Based Maritime Vessel Inspection: A Perception-Driven Navigation Strategy"** (Under Review in *Ocean Engineering*).

## 🌟 Key Highlights
- **The "Data Augmentation Trap":** We discovered and resolved the systemic "Lateral (Port-Starboard) Confusion" in maritime vision by deliberately eliminating physics-violating horizontal flip augmentations.
- **Extreme Edge Benchmarking:** Evaluated 50 YOLO architectures (v8-v12) on an NVIDIA Jetson Orin Nano Super (8GB) under strict **60-second sustained thermal equilibrium loads**.
- **Adaptive WNS Protocol:** Proposed a Weighted Normalized Scoring (WNS) system, identifying **`YOLOv9t-engine`** as the Pareto-optimal edge model (0.879 mAP@50, 72.78 FPS, 5.98 W).
- **Sequence-Level Simulation:** Quantitatively proved that our Perception-Initialized Guided Search Strategy (Initial-GSS) reduces the UAV target search time by **44.7%** compared to unguided methods.

---

## 📂 Repository Structure

```text
UAV-Maritime-Inspection/
│
├── dataset/                  
│   └── dataset_link.md             # Contains links and instructions for the dataset
│
├── videos/                   
│   ├── video_4_fps_power.mp4 # Pre-loaded video for strict GPU load testing
│   └── video_4_sequence_eval.mp4 # High-fidelity 3D circumnavigation for GSS evaluation
│
├── scripts/                  
│   ├── train_custom.sh                     # Bash script for training and TensorRT export (w/o fliplr)
│   ├── auto_benchmark.py                   # Automated 60s thermal equilibrium & power test script
│   ├── verify_wns.py                       # Script to calculate WNS rankings and normalization
│   ├── sequence_once.py                    # Core logic for single Monte Carlo trial trajectory
│   ├── sequence_simulation.py              # Execution script for massive Monte Carlo runs
│   └── sequence_once_predictions_cache.xlsx# The input file of sequence_once.py
│
├── logs/                     
│   ├── tegrastats_logs/      # Raw VDD_IN power consumption logs
│   ├── simulate_logs/        # Raw sequence_eval simulation flight logs
│   ├── benchmark_results_final.csv # Benchmark metrics for all 50 models
│   └── Appendix_Model_Metrics.csv      # Extended metrics (Params, GFLOPs, Precision, etc.)
│
└── README.md                 
```

---

## 🛠️ Hardware & Software Requirements

To perfectly reproduce the edge evaluation environments presented in the paper, the following setup is strictly required:
- **Hardware:** NVIDIA Jetson Orin Nano Super (8GB Memory).
- **OS:** Native Ubuntu 22.04.4 LTS (JetPack 6.x).
- **Docker Environment:** `ultralytics/ultralytics:latest-jetson-jetpack6` (Docker engine version < 28.0 required to avoid JetPack runtime conflicts).

---

## 🚀 Quick Start / Reproducibility Guide

### 1. Data & Pre-trained Models Access
We have open-sourced all necessary data and optimized computational graphs:
- 🛳️ **Vessel Perspectives Dataset (1,967 images, 5 classes):** [Download via Google Drive](https://drive.google.com/file/d/1rm11-l1lWx8frvlJWpTZ8psUeAqdbzrG/view?usp=sharing)
- 🧠 **All 50 Models (PyTorch `.pt` & TensorRT `.engine`):** [Download via Google Drive](https://drive.google.com/file/d/1FEBnxlvqQmmZ7aioJIL1_O2xZLTWDVmc/view?usp=sharing)

### 2. Training and TensorRT Exporting
To avoid the "Lateral Confusion" trap, horizontal flipping must be disabled (`fliplr=0.0`). We also enforce FP16 precision (`half=True`), a fixed 4GB workspace, and `batch=1` to guarantee edge determinism. 
Run the integrated bash script:
```bash
# This script handles both training and subsequent TensorRT engine generation
bash scripts/train_custom.sh
```

### 3. 60-Second Thermal Equilibrium Benchmark
Run the automated benchmarking script inside the Jetson Docker container. This script pre-loads `video_4_fps_power.mp4` into RAM to bypass I/O bottlenecks and records `VDD_IN` power via `tegrastats` for 60 seconds per model.
```bash
python scripts/auto_benchmark.py
```

### 4. WNS Scoring & Sequence-Level Simulation
Calculate the WNS ranking and run the timeline-based Monte Carlo simulation on `video_4_sequence_eval.mp4` to quantitatively validate the flight endurance gains.
```bash
# Verify the dynamic ranking across 4 extreme operational scenarios
python scripts/verify_wns.py

# Run the Initial-GSS vs BSS endurance evaluation (10,000 trials)
python scripts/sequence_simulation.py
```

---

## 📜 Citation
If you find this repository, dataset, or the architectural insights useful for your research, please consider citing our paper:
```bibtex
@article{cao2026guided,
  title={Guided Search and Edge YOLO Benchmarking for UAV-Based Maritime Vessel Inspection: A Perception-Driven Navigation Strategy},
  author={Cao, Zhi-Bo},
  journal={Ocean Engineering},
  year={2026},
  note={Under Review}
}
