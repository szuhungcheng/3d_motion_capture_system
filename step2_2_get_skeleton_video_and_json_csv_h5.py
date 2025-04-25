import os
import sys
import subprocess
import shutil
import json
import csv
import pandas as pd
from scipy.signal import butter, filtfilt
import cv2
import time

print("opencv_version", cv2.__version__)

# Set AlphaPose directory as working directory
alphapose_dir = r"C:\AlphaPose"
checkpoint_path = r"C:\AlphaPose\pretrained_models\pose_model.pth"
config_path = r"C:\AlphaPose\configs\coco\resnet\256x192_res50_lr1e-3_1x.yaml"
os.chdir(alphapose_dir)
sys.path.append(alphapose_dir)

base_video_dir = r"C:\Users\USER\OneDrive\Desktop_20250211_after_reset\康舒_motion\recorded_videos"
skeleton_video_dir = r"C:\Users\USER\OneDrive\Desktop_20250211_after_reset\康舒_motion\skeleton_videos"
json_dir = r"C:\Users\USER\OneDrive\Desktop_20250211_after_reset\康舒_motion\joints_json_files"
csv_dir = r"C:\Users\USER\OneDrive\Desktop_20250211_after_reset\康舒_motion\joints_csv_files"
h5_dir = r"C:\Users\USER\OneDrive\Desktop_20250211_after_reset\康舒_motion\joints_h5_files"

# --- 初始化資料夾結構 ---
output_dirs = {
    "videos": base_video_dir,
    "skeleton": skeleton_video_dir,
    "json": json_dir,
    "csv": csv_dir,
    "h5": h5_dir
}

for name, path in output_dirs.items():
    os.makedirs(path, exist_ok=True)
    print(f"📁 {name} 資料夾準備完成：{path}")



def build_output_paths_from_video(video_path):
    filename = os.path.basename(video_path)
    folder_name = os.path.splitext(filename)[0]
    output_paths = {
        "skeleton_video": os.path.join(skeleton_video_dir, folder_name),
        "json": os.path.join(json_dir, folder_name),
        "csv": os.path.join(csv_dir, folder_name),
        "h5": os.path.join(h5_dir, folder_name),
    }
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)
    return output_paths

def find_all_avi_files(base_dir):
    avi_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".avi"):
                full_path = os.path.join(root, file)
                avi_files.append(full_path)
    return avi_files

video_paths_list = find_all_avi_files(base_video_dir)
print(f"✅ 共找到 {len(video_paths_list)} 個影片：")

# --- Run AlphaPose on each video ---
for video_path in video_paths_list:
    output_file_name = os.path.splitext(os.path.basename(video_path))[0]
    output_json_path = os.path.join(json_dir, f"AlphaPose_{output_file_name}")

    command = [
        "python", os.path.join(alphapose_dir, "scripts", "demo_inference.py"),
        "--video", video_path,
        "--outdir", output_json_path,
        "--checkpoint", checkpoint_path,
        "--cfg", config_path,
        "--save_video",
        "--vis_fast",
        "--gpus", "0"
    ]

    try:
        print(f"\n🚀 Running AlphaPose on: {video_path}")
        subprocess.run(command, check=True)

        if os.path.exists(output_json_path):
            print(f"✅ JSON folder created: {output_json_path}")
        else:
            print(f"⚠️ JSON result folder not found for: {output_json_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ AlphaPose error on {video_path}: {e}")
        continue


"""
搬遷影片和 JSON 檔案到指定資料夾
並刪除原始資料夾
"""


# --- 初始化資料夾結構 ---
output_dirs = {
    "videos": base_video_dir,
    "skeleton": skeleton_video_dir,
    "json": json_dir,
    "csv": csv_dir,
    "h5": h5_dir
}

for name, path in output_dirs.items():
    os.makedirs(path, exist_ok=True)
    print(f"📁 {name} 資料夾準備完成：{path}")

print("\n📦 開始整理輸出資料夾...")
failed_folders = []

for folder_name in os.listdir(json_dir):
    folder_path = os.path.join(json_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    print(f"\n🔍 處理資料夾：{folder_path}")
    base_name = folder_name.replace("AlphaPose_output_", "")
    json_src = os.path.join(folder_path, "alphapose-results.json")
    json_dst = os.path.join(json_dir, f"AlphaPose_output_{base_name}.json")

    # 預期的 avi 檔名
    expected_avi = os.path.join(folder_path, f"{folder_name}.avi")
    avi_dst = os.path.join(skeleton_video_dir, f"AlphaPose_output_{base_name}.avi")

    # 檢查 avi 存在與否，若找不到就 fallback 掃描所有 avi
    avi_src = None
    if os.path.exists(expected_avi):
        avi_src = expected_avi
        print(f"✅ 找到預期 avi：{avi_src}")
    else:
        print(f"⚠️ 找不到預期 avi：{expected_avi}，開始掃描資料夾內容...")
        for f in os.listdir(folder_path):
            if f.lower().endswith(".avi"):
                avi_src = os.path.join(folder_path, f)
                print(f"🔁 找到其他 avi：{avi_src}")
                break

    try:
        if avi_src and os.path.exists(avi_src) and os.path.getsize(avi_src) > 0:
            os.rename(avi_src, avi_dst)
            print(f"🎥 移動 avi 到：{avi_dst}")
        else:
            print(f"⚠️ 無法處理 avi：{avi_src}（不存在或為空）")

        if os.path.exists(json_src):
            os.rename(json_src, json_dst)
            print(f"📄 移動 JSON 到：{json_dst}")
        else:
            print(f"⚠️ 找不到 JSON：{json_src}")

        time.sleep(0.5)
        shutil.rmtree(folder_path)
        print(f"🧹 已刪除資料夾：{folder_path}")

    except Exception as e:
        print(f"❌ 無法處理：{folder_path}，錯誤：{e}")
        failed_folders.append(folder_path)







"""
轉json到csv
"""
coco_keypoints = [
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip",
    "LKnee", "RKnee", "LAnkle", "RAnkle"
]

# --- Apply Butterworth Filter ---
def butterworth_filter(data, cutoff=3, fs=30, order=2):
    if len(data) < order * 3:
        return data
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# --- 轉換所有 JSON 檔為 CSV ---
print("\n📄 開始轉換 JSON 為 CSV...")

for file in os.listdir(json_dir):
    if not file.endswith(".json"):
        continue

    json_path = os.path.join(json_dir, file)
    csv_path = os.path.join(csv_dir, file.replace(".json", ".csv"))
    print(f"\n🔍 處理檔案：{json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            print(f"⚠️ 空 JSON：{json_path}")
            continue

        # 建立一個 dict：key 為 joint 名，值為 (x序列, y序列)
        keypoint_series = {kp: {"x": [], "y": []} for kp in coco_keypoints}

        for item in data:
            kp_list = item["keypoints"]
            for i, kp in enumerate(coco_keypoints):
                x = kp_list[i * 3]
                y = kp_list[i * 3 + 1]
                keypoint_series[kp]["x"].append(x)
                keypoint_series[kp]["y"].append(y)

        # 濾波處理後建立 dataframe
        filtered_data = {}
        for kp in coco_keypoints:
            filtered_x = butterworth_filter(keypoint_series[kp]["x"])
            filtered_y = butterworth_filter(keypoint_series[kp]["y"])
            filtered_data[f"{kp}_x"] = filtered_x
            filtered_data[f"{kp}_y"] = filtered_y

        df = pd.DataFrame(filtered_data)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已輸出 CSV：{csv_path}")

    except Exception as e:
        print(f"❌ 處理失敗：{json_path}，錯誤：{e}")




"""
轉json到h5
"""
print("\n📁 開始轉換 JSON 為 HDF5 檔案...")

# 定義 COCO keypoints 順序與 HDF5 MultiIndex 結構

scorer = "AlphaPose"
coords = ["x", "y", "likelihood"]
multi_index = pd.MultiIndex.from_product([[scorer], coco_keypoints, coords], names=["scorer", "bodyparts", "coords"])



# 處理每個 JSON 檔案
for json_file in os.listdir(json_dir):
    if not json_file.endswith(".json"):
        continue

    json_path = os.path.join(json_dir, json_file)
    h5_output_path = os.path.join(h5_dir, json_file.replace(".json", "_filtered.h5"))

    print(f"\n🔍 處理檔案：{json_path}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            print(f"⚠️ 空 JSON：{json_path}")
            continue

        keypoints_list = [entry["keypoints"] for entry in data]
        num_joints = len(coco_keypoints)

        # 基本驗證
        assert len(keypoints_list[0]) == num_joints * 3, f"❌ JSON 結構錯誤：{json_file}"

        # 建立 DataFrame（MultiIndex）
        df = pd.DataFrame(keypoints_list, columns=multi_index)

        # 套用濾波
        for kp in coco_keypoints:
            for axis in ["x", "y"]:
                df[(scorer, kp, axis)] = butterworth_filter(df[(scorer, kp, axis)])

        # 寫入 HDF5
        df.to_hdf(h5_output_path, key="df_with_missing", mode="w")
        print(f"✅ 已儲存 HDF5：{h5_output_path}")

    except Exception as e:
        print(f"❌ 轉換失敗：{json_path}，錯誤：{e}")

print("\n🎉 所有 JSON 已轉換為 HDF5！")


