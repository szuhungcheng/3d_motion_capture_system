import cv2
import os
import datetime
import threading
import time
import re
import gc
import json

"""
record_video_0:chess_board
record_video_1:neck side bending
record_video_2:neck flexion/extension
record_video_3:shoulder elevation
record_video_4:trunk side bending
record_video_5:trunk flexion/extension
record_video_6:DSWAO
"""
video_dictionary = {
    "0": "chess_board",
    "1": "neck_side_bending",
    "2": "neck_flexion/extension",
    "3": "shoulder_elevation",
    "4": "trunk_side_bending",
    "5": "trunk_flexion/extension",
    "6": "DSWAO"
    }


print("OpenCV version:", cv2.__version__)
# Check available cameras
print("Checking available camera indexes...")
camera_indices = []

for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"✅ Camera detected at index {i}")
        camera_indices.append(i)
    else:
        print(f"❌ No camera found at index {i}")
    cap.release()

if not camera_indices:
    print("⚠ No cameras detected. Exiting...")
    exit()


def sanitize_filename(name):
    return re.sub(r'[\\/:"*?<>|]+', "_", name)

def choose_video_type_and_id():
    pid = input("enter subject number（example: 1）：")
    person_id = f"person_{pid.zfill(3)}"
    print("enter the type of videos for recording:\n0=chessboard\n1=neck side bending\n2=trunk side bending\n3=shoulder elevation\n4=DSWAO\n5=neck flexion/extension\n6=trunk flexion/extension")
    k = input()
    if k not in video_dictionary:
        print("⚠ failed input, restart the code")
        exit()
    else:
        print("the type of recording videos is:", video_dictionary[k])
    
    return video_dictionary[k],person_id 
#返回一個影片類型變數和受測者編號變數


def choose_video_type_only():
    print("enter the type of videos for recording:\n0=chessboard\n1=neck side bending\n2=trunk side bending\n3=shoulder elevation\n4=DSWAO\n5=neck flexion/extension\n6=trunk flexion/extension")
    k = input()
    if k not in video_dictionary:
        print("⚠ failed input, restart the code")
        exit()
    else:
        print("the type of recording videos is:", video_dictionary[k])
    return video_dictionary[k] 
#返回一個影片類型變數

def start_recording(vt, person_id):
    safe_vt = sanitize_filename(vt)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_folder = os.path.join("recorded_videos", safe_vt)
    folder_name = f"{person_id}_{safe_vt}_{timestamp}"
    save_folder = os.path.join(base_folder, folder_name)
    os.makedirs(save_folder, exist_ok=True)
    print("📁 create the svae folder：", save_folder)

    recording_flag = threading.Event()
    caps = [cv2.VideoCapture(i, cv2.CAP_DSHOW) for i in camera_indices]
    print("caps:\n", caps)

    video_filenames_list = [os.path.join(save_folder, f"output_{person_id}_{safe_vt}_cam{i+1}_{timestamp}.avi") for i in range(len(camera_indices))]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    for cap in caps:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    frame_counts_list = [0] * len(camera_indices)  # ← 預先建立固定長度 list

    def record_video(cap, filename, index):
        ret, frame = cap.read()
        height, width, _ = frame.shape
        print(f"📹 Camera {cap} recording at {width}x{height}")

        out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
        start_time_ns = time.time_ns()
        frame_count = 0

        while recording_flag.is_set():
            ret, frame = cap.read()
            out.write(frame)
            frame_count += 1

        frame_counts_list[index] = frame_count  # ← 放到對應 index
        cap.release()
        out.release()

        print(f"📁 Recording stopped")
        print(f"🕒 Start time: {datetime.datetime.fromtimestamp(start_time_ns / 1e9).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        print(f"🎞 Frame count: {frame_count}")

    cv2.namedWindow("Webcam concatenate", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam concatenate", 1500, 400)
    print("🎥 Press 'r' to start recording, 's' to stop.")

    threads = []

    while True:
        frames = []

        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (640, 480))
                line_color = (0, 255, 255)
                line_thickness = 2
                height, width, _ = frame_resized.shape
                mid_x = width // 2
                cv2.line(frame_resized, (mid_x, 0), (mid_x, height), line_color, line_thickness)
                frames.append(frame_resized)
            else:
                frames.append(None)

        valid_frames = [f for f in frames if f is not None]
        if valid_frames:
            combined_frame = cv2.hconcat(valid_frames) if len(valid_frames) > 1 else valid_frames[0]
            cv2.imshow("Webcam concatenate", combined_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and not recording_flag.is_set():
            print("🔴 Recording")
            recording_flag.set()
            threads = []
            for i, (cap, filename) in enumerate(zip(caps, video_filenames_list)):
                thread = threading.Thread(target=record_video, args=(cap, filename, i))
                threads.append(thread)
                thread.start()

        elif key == ord('s') and recording_flag.is_set():
            print("🛑 Stopping")
            recording_flag.clear()
            for thread in threads:
                thread.join()

            entry = {
                "person_id": person_id,
                "folder_name": folder_name,
                "timestamp": timestamp,
                "camera_count": len(camera_indices),
                "video_files": video_filenames_list,
                "frame_counts": frame_counts_list,  # ✅ 新增這一行
                "resolution": "1920x1080",
                "fps": 30,
            }

            metadata_path = os.path.join(base_folder, "metadata.json")

            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    try:
                        metadata_list = json.load(f)
                    except json.JSONDecodeError:
                        print("⚠ metadata.json was broken, please restart")
                        metadata_list = []
            else:
                metadata_list = []

            metadata_list.append(entry)

            with open(metadata_path, "w") as f:
                json.dump(metadata_list, f, indent=4)

            break

    for cap in caps:
        cap.release()
        del cap
    gc.collect()
    cv2.destroyAllWindows()





vt,person_id = choose_video_type_and_id()
start_recording(vt,person_id)

while True:
    print("record same subject or not？(y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        vt2= choose_video_type_only()
        start_recording(vt2,person_id)
    elif choice == 'n':
        print(person_id+",recording finish")
        break
    else:
        print("⚠failed input, 'y' or 'n' only 'y' 或 'n'。")
