# 3D Motion Capture System: From Webcam to 3D Biomechanical Data

# Statement
This repository presents a modular 3D motion capture system that uses three webcams to perform full-body 3D joint reconstruction, leveraging state-of-the-art tools including [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) for 2D keypoint detection and [Anipose](https://anipose.org/) for multi-view triangulation.


---
## 📁 Project Content
---

### `step1_get_video_webcam_version_3_camera.py`

Records synchronized videos from **three USB webcams** using parallel threads. Press `r` to begin recording.

> ⚠ Frame drop may occur based on your hardware I/O capacity — USB 3.0 preferred.

---

### `step2_1_get_calibration_file.py`

Captures relative positions between all cameras with a printed **chessboard** visible to all cameras. You must modify parameters according to your chessboard size (e.g., square size in cm, rows/columns count), or you can use the attached chessboard.jpg and preset parameters in this repository.

---

### `step2_2_get_skeleton_video_and_json_csv.py`

This script wraps the **AlphaPose** detector. You must:
- download AlphaPose manually: https://github.com/MVIG-SJTU/AlphaPose
- Set the appropriate local paths inside this script
- Provide input videos captured from step 1

It outputs:
- 2D Overlaid skeleton video clips
- 2D `.json` and `.csv` joint coordinates
- 2D `.h5` file formatted for Anipose (vital for 3D reconstruction)

---

### `step3_get_3d_data.py`

Uses **Anipose**'s triangulation pipeline to reconstruct 3D motion data by aligning multi-view 2D joint detections via epipolar geometry.

- Uses camera calibration (`camera_calibration.yaml`)
- Input: `.h5` joint data
- Output: `.csv` file containing 3D data for each joint across time

Make sure your camera order and calibration labels are consistent.

---

### `step4_visualization.py`

Uses plotly to visualize and check 3D data quickly 



## 🔧 System Requirements

- **Python**: 3.9
- **Hardware**: 
  - Laptop with sufficient USB bandwidth
  - Three standard USB webcams (e.g., Logitech C920)
  - Decent GPU (optional but recommended for AlphaPose inference)

---


## 🚦 Instructions

### 🔴 Step 1 — Webcam Video Recording  
`step1_get_video_webcam_version_3_camera.py`

Records synchronized videos from **three USB webcams** using parallel threads. You would get synchronized videos in different folders.

> ⚠ Frame drop may occur based on your hardware I/O capacity — USB 3.0 preferred.

---

### 🧩 Step 2.1 — Camera Calibration  
`step2_1_get_calibration_file.py`

Use several chessboard videos to get the extrinsic parameters (relative positions and directions between cameras). The output here would be `calibration.toml`, a file contained intrinsic and extrinsic parameters between cameras. We suggest that the error from the matrix lower than 0.3 is appropriated.
---

### 🧍 Step 2.2 — 2D Joint Detection (AlphaPose)  
`step2_2_get_skeleton_video_and_json_csv.py`

Detect 2D joints for each videos. output file have several types in distinct folder: skeleton videos, json files, csv files, and h5 files about 2d joint positions

---

### 🧊 Step 3 — 3D Reconstruction with Anipose  
`step3_get_3d_data.py`

Integrate same set of 2D h5 files for triangulation to reconstruct 3D motion data by aligning multi-view 2D joint detections via parameters in `calibration.toml`. You would get 3D joint position CSV files for each set of 2D data.



Make sure your camera fixed and calibration labels are consistent in the real chessboard and size in the code.

---


## 📜 License & Credits

- **System Code**: MIT License
- **2D Pose**: [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose), released under their license
- **3D Reconstruction**: [Anipose](https://github.com/lambdaloop/anipose), free and open-source

---

## 👤 Author

Developed by [Cheng, Szu-Hung](https://github.com/Mars-Zheng)
Sports Science Engineer | Biomechanics Strategist | Boxing Athlete  
Passionate about fusing **ecological psychology**, **movement intelligence**, and **technology**.

---

## 📫 Contact

If you’re a researcher, coach, or company interested in customized motion analysis systems, feel free to open an issue or reach out via email: eric14209@gmail.com.

