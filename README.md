# 3D Motion Capture System: From Webcam to 3D Biomechanical Analysis

This repository presents a modular 3D motion capture system that uses three consumer-grade webcams to perform full-body 3D joint reconstruction, leveraging state-of-the-art tools including [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) for 2D keypoint detection and [Anipose](https://anipose.org/) for multi-view triangulation.

Designed with clinical and sports applications in mind, this system enables researchers and practitioners to:

- Efficiently record synchronized video data across three cameras
- Perform accurate intrinsic/extrinsic camera calibration
- Detect 2D joint keypoints robustly using pretrained deep learning models
- Reconstruct 3D motion trajectories from multi-view 2D joint data
- Analyze joint motion for clinical and sports performance assessment
- Generate interactive 3D animations for visualization and feedback

---

## 📁 Project Structure

---

## 🔧 System Requirements

- **Python**: 3.9–3.11
- **Hardware**: 
  - Laptop with sufficient USB bandwidth
  - Three standard USB webcams (e.g., Logitech C920)
  - Decent GPU (optional but recommended for AlphaPose inference)

---

## 🚦 Pipeline Overview

### 🔴 Step 1 — Webcam Video Recording  
`step1_get_video_webcam_version_3_camera.py`

Records synchronized videos from **three USB webcams** using parallel threads. Press `r` to begin recording.

> ⚠ Frame drop may occur based on your hardware I/O capacity — USB 3.0 preferred.

---

### 🧩 Step 2.1 — Camera Calibration  
`step2_1_get_calibration_file.py`

Captures relative positions between all cameras with a printed **chessboard** visible to all cameras. The script extracts calibration frames. You must modify parameters according to your chessboard size (e.g., square size in cm, rows/columns count).

---

### 🧍 Step 2.2 — 2D Joint Detection (AlphaPose)  
`step2_2_get_skeleton_video_and_json_csv.py`

This script wraps the **AlphaPose** detector. You must:
- Clone AlphaPose manually: https://github.com/MVIG-SJTU/AlphaPose
- Set the appropriate local paths inside this script
- Provide input videos captured from step 1

It outputs:
- Overlaid skeleton video clips
- `.json` and `.csv` joint coordinates
- `.h5` file formatted for Anipose (vital for 3D reconstruction)

AlphaPose is a deep learning–based top-down pose estimator, with strong robustness for human body joints under varying conditions.

---

### 🧊 Step 3 — 3D Reconstruction with Anipose  
`step3_get_3d_data.py`

Uses **Anipose**'s triangulation pipeline to reconstruct 3D motion trajectories by aligning multi-view 2D joint detections via epipolar geometry.

- Uses camera calibration (`camera_calibration.yaml`)
- Input: `.h5` joint data
- Output: `.csv` file containing 3D coordinates for each joint across time

Make sure your camera order and calibration labels are consistent.

---

### 📊 Step 4 — Parameter Analysis  
`step4_analysis_parameters_calcualte.py`

This script computes **key biomechanical or clinical parameters** from the 3D data, including:

- Joint range of motion (ROM)
- Displacement symmetry
- Sway patterns
- Temporal events (e.g., peak velocity timing)

Parameters are selected in consultation with a licensed physical therapist.

---

### 🎥 Step 5 — Visualization  
`step5_visualization.py`

Generates a **3D animation in HTML format** using Plotly or similar. You can open the animation in any modern web browser and explore the 3D joint trajectories interactively.

Used for both:
- Internal validation (debugging geometry or timing)
- External reports or feedback (clinical or athlete review)

---

## 🧪 Example Use Case

Imagine you’re analyzing gait stability in patients post-stroke or optimizing lower-limb coordination for boxers. This system lets you:
- Record a sparring or rehab session from 3 angles
- Detect motion with AlphaPose
- Reconstruct 3D dynamics
- Quantify asymmetry or instability
- Provide animated feedback to patients/athletes

---

## 📂 Recommended Folder Layout


---

## 📌 Notes and Best Practices

- Use **tripods** to keep cameras static.
- Avoid occlusions in camera placement.
- Sync camera clocks or record visual clap for alignment.
- Lighting consistency matters — AlphaPose is good, but not magic.

---

## 📜 License & Credits

- **System Code**: MIT License
- **2D Pose**: [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), released under their license
- **3D Reconstruction**: [Anipose](https://github.com/lambdaloop/anipose), free and open-source

---

## 👤 Author

Developed by [Cheng, Szu-Hung](https://github.com/Mars-Zheng),  
Sports Science Engineer | Biomechanics Strategist | Boxing Athlete  
Passionate about fusing **ecological psychology**, **movement intelligence**, and **technology**.

---

## 📫 Contact

If you’re a researcher, coach, or company interested in customized motion analysis systems, feel free to open an issue or reach out via [email](eric14209@gmail.com).

