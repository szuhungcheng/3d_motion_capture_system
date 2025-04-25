import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.signal import butter, filtfilt
from aniposelib.cameras import CameraGroup
from aniposelib.utils import load_pose2d_fnames

# --- SETTINGS ---
h5_folder = r'C:\Users\USER\OneDrive\Desktop_20250211_after_reset\康舒_motion\joints_h5_files'
calibration_file = 'calibration.toml'
output_folder = 'output_3d_data'
score_threshold = 0.5

# --- FILTER FUNCTION ---
def butter_lowpass_filter(data, cutoff=3, fs=30, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0, padlen=10)

# --- GROUPING H5 FILES ---
def group_h5_files(h5_dir):
    pattern = re.compile(
        r'AlphaPose_output_(person_\d+)_(.+?)_(cam\d)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_filtered)?\.h5'
    )

    cam_mapping = {
        'cam1': 'A',
        'cam2': 'B',
        'cam3': 'C'
    }

    groups = defaultdict(dict)

    for fname in os.listdir(h5_dir):
        match = pattern.match(fname)
        if match:
            pid, movement, cam_key = match.groups()
            cam_name = cam_mapping.get(cam_key)
            if cam_name:
                key = (pid, movement)
                groups[key][cam_name] = os.path.join(h5_dir, fname)
            else:
                print(f"[!] Unknown camera key '{cam_key}' in filename: {fname}")
        else:
            print(f"[!] Filename doesn't match pattern: {fname}")
    return groups

# --- PROCESS EACH TRIO ---
def process_group(fname_dict, cgroup, output_csv):
    d = load_pose2d_fnames(fname_dict, cam_names=cgroup.get_names())
    n_cams, n_frames, n_joints, _ = d['points'].shape
    points = d['points']
    scores = d['scores']
    bodyparts = d['bodyparts']

    # Drop low-confidence points
    points[scores < score_threshold] = np.nan

    points_flat = points.reshape(n_cams, -1, 2)
    p3ds_flat = cgroup.triangulate(points_flat, progress=False)
    p3ds = p3ds_flat.reshape(n_frames, n_joints, 3)

    columns = ["Frame"] + [f"{bp}_{axis}" for bp in bodyparts for axis in ["X", "Y", "Z"]]
    data = []
    for frame in range(n_frames):
        row = [frame]
        for joint in range(n_joints):
            row.extend(p3ds[frame, joint, :])
        data.append(row)

    df = pd.DataFrame(data, columns=columns)

    # Apply Butterworth filter
    for col in df.columns[1:]:
        valid_idx = ~df[col].isna()
        if valid_idx.sum() > 5:
            df.loc[valid_idx, col] = butter_lowpass_filter(df.loc[valid_idx, col].values)

    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved 3D CSV: {output_csv}")

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("📡 Loading camera calibration...")
    cgroup = CameraGroup.load(calibration_file)

    print("🔍 Scanning for valid H5 groups...")
    groups = group_h5_files(h5_folder)

    print(f"📦 Found {len(groups)} subject-movement pairs.\n")

    for (person, movement), cams in groups.items():
        if all(k in cams for k in ['A', 'B', 'C']):
            fname_dict = {
                'A': cams['A'],
                'B': cams['B'],
                'C': cams['C'],
            }
            output_csv = os.path.join(output_folder, f"{person}_{movement}_3d.csv")
            process_group(fname_dict, cgroup, output_csv)
        else:
            print(f"[!] Skipping {person}-{movement}: missing some camera views (found {list(cams.keys())})")
