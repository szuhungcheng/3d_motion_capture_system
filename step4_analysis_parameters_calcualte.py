import os
import re
import math
import csv
import pandas as pd

INPUT_FOLDER = 'output_3d_data'
OUTPUT_FOLDER = 'analysis_raw_data'
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, 'summary_analysis.csv')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def vector(p1, p2):
    return [p2[i] - p1[i] for i in range(3)]

def length(v):
    return math.sqrt(sum([x**2 for x in v]))

def dot(v1, v2):
    return sum([v1[i] * v2[i] for i in range(3)])

def angle_between(pA, pB, pC):
    try:
        BA = vector(pB, pA)
        BC = vector(pB, pC)
        if length(BA) * length(BC) < 1e-6:
            return None
        cos_theta = dot(BA, BC) / (length(BA) * length(BC))
        cos_theta = max(min(cos_theta, 1.0), -1.0)
        return math.degrees(math.acos(cos_theta))
    except:
        return None

def midpoint(p1, p2):
    return [(p1[i] + p2[i]) / 2.0 for i in range(3)]

def distance(p1, p2):
    try:
        return length(vector(p1, p2))
    except:
        return None

class Pose3DCSV:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self.filename = os.path.basename(filepath)
        self.subject, self.movement = self.parse_filename()

    def parse_filename(self):
        match = re.match(r'person_(\d+)_(.*?)_3d\.csv', self.filename)
        return f"person_{match.group(1)}", match.group(2).lower()

    def get_point(self, joint, frame):
        try:
            p = [self.df[f"{joint}_{axis}"].iloc[frame] for axis in ['X', 'Y', 'Z']]
            if any(pd.isna(v) for v in p):
                return None
            return p
        except:
            return None

    def safe_series(self, values):
        return [v for v in values if v is not None]

    def analyze(self):
        move = self.movement
        subject = self.subject
        results = []

        def record(cat, move, value, type_desc, timing_desc):
            if value is not None and not pd.isna(value):
                results.append((cat, move, round(value, 3), subject, type_desc, timing_desc))

        df = self.df
        N = len(df)

        def pt(j, i): return self.get_point(j, i)
        def mid(a, b, i): return midpoint(pt(a, i), pt(b, i)) if pt(a, i) and pt(b, i) else None

        if move == "neck_side_bending":
            cat = "上肢"
            angles = [angle_between(pt("LShoulder", i), pt("Nose", i), pt("RShoulder", i)) for i in range(N)]
            clean = self.safe_series(angles)
            if clean:
                record(cat, move, min(clean), "左側脖子夾角最小值", "NaN")
                record(cat, move, max(clean), "右側脖子夾角最小值", "NaN")

        elif move == "shoulder_elevation":
            cat = "上肢"
            d1 = [distance(pt("LShoulder", i), pt("LHip", i)) for i in range(N)]
            d2 = [distance(pt("RShoulder", i), pt("RHip", i)) for i in range(N)]
            d3 = [distance(pt("LShoulder", i), pt("Nose", i)) for i in range(N)]
            d4 = [distance(pt("RShoulder", i), pt("Nose", i)) for i in range(N)]

            # 計算左肩到右肩的距離
            shoulder_width = [distance(pt("LShoulder", i), pt("RShoulder", i)) if pt("LShoulder", i) and pt("RShoulder", i) else None for i in range(N)]
            def normalized(val, i): return val / shoulder_width[i] if val and shoulder_width[i] else None

            if (s := self.safe_series(d1)):
                idx = d1.index(max(s))
                if (val := normalized(d1[idx], idx)): record(cat, move, val, "左肩至左髖距離最大值", "NaN")
            if (s := self.safe_series(d2)):
                idx = d2.index(max(s))
                if (val := normalized(d2[idx], idx)): record(cat, move, val, "右肩至右髖距離最大值", "NaN")
            if (s := self.safe_series(d3)):
                idx = d3.index(max(s))
                if (val := normalized(d3[idx], idx)): record(cat, move, val, "左肩至鼻子距離最大值", "NaN")
            if (s := self.safe_series(d4)):
                idx = d4.index(max(s))
                if (val := normalized(d4[idx], idx)): record(cat, move, val, "右肩至鼻子距離最大值", "NaN")

        elif move == "neck_flexion_extension":
            cat = "上肢"
            angles = []
            for i in range(N):
                sh = mid("LShoulder", "RShoulder", i)
                hi = mid("LHip", "RHip", i)
                if pt("Nose", i) and sh and hi:
                    angles.append(angle_between(pt("Nose", i), sh, hi))
                else:
                    angles.append(None)
            clean = self.safe_series(angles)
            if clean:
                record(cat, move, max(clean), "三點夾角最大值", "NaN")
                record(cat, move, min(clean), "三點夾角最小值", "NaN")

        elif move == "trunk_side_bending":
            cat = "軀幹"
            angles = []
            for i in range(N):
                sh = mid("LShoulder", "RShoulder", i)
                hi = mid("LHip", "RHip", i)
                kn = mid("LKnee", "RKnee", i)
                if sh and hi and kn:
                    angles.append(angle_between(sh, hi, kn))
                else:
                    angles.append(None)
            clean = self.safe_series(angles)
            if clean:
                record(cat, move, max(clean), "肩髖膝中點夾角最大值", "左側 side bending")
                record(cat, move, min(clean), "肩髖膝中點夾角最小值", "右側 side bending")

        elif move == "trunk_flexion_extension":
            cat = "下肢"
            shk, hka = [], []
            for i in range(N):
                sh = mid("LShoulder", "RShoulder", i)
                hi = mid("LHip", "RHip", i)
                kn = mid("LKnee", "RKnee", i)
                an = mid("LAnkle", "RAnkle", i)
                shk.append(angle_between(sh, hi, kn) if sh and hi and kn else None)
                hka.append(angle_between(hi, kn, an) if hi and kn and an else None)
            clean_shk = self.safe_series(shk)
            if clean_shk:
                max_idx = shk.index(max(clean_shk))
                min_idx = shk.index(min(clean_shk))
                record(cat, move, shk[max_idx], "肩髖膝夾角最大值", "NaN")
                record(cat, move, hka[max_idx], "髖膝踝夾角", "肩髖膝夾角最大值時")
                record(cat, move, shk[min_idx], "肩髖膝夾角最小值", "NaN")
                record(cat, move, hka[min_idx], "髖膝踝夾角", "肩髖膝夾角最小值時")
                clean_hka = self.safe_series(hka)
                if clean_hka:
                    record(cat, move, clean_hka[0], "髖膝踝夾角 (benchmarking)", "站直時")

        elif move == "dswao":
            cat = "下肢"
            angles, lk_d, rk_d = [], [], []
            nose_l, nose_r = [], []

            # 加入每幀的髖寬距離
            hip_span = [distance(pt("LHip", i), pt("RHip", i)) if pt("LHip", i) and pt("RHip", i) else None for i in range(N)]
            def normalized(val, i): return val / hip_span[i] if val and hip_span[i] else None

            for i in range(N):
                sh = mid("LShoulder", "RShoulder", i)
                hi = mid("LHip", "RHip", i)
                kn = mid("LKnee", "RKnee", i)
                angles.append(angle_between(sh, hi, kn) if sh and hi and kn else None)
                lk_d.append(distance(pt("LKnee", i), hi) if pt("LKnee", i) and hi else None)
                rk_d.append(distance(pt("RKnee", i), hi) if pt("RKnee", i) and hi else None)

                # 鼻肩肘角
                nose_l.append(angle_between(pt("Nose", i), pt("LShoulder", i), pt("LElbow", i)) if pt("Nose", i) and pt("LShoulder", i) and pt("LElbow", i) else None)
                nose_r.append(angle_between(pt("Nose", i), pt("RShoulder", i), pt("RElbow", i)) if pt("Nose", i) and pt("RShoulder", i) and pt("RElbow", i) else None)

            clean = self.safe_series(angles)
            if clean:
                max_idx = angles.index(max(clean))
                min_idx = angles.index(min(clean))

                record(cat, move, max(clean), "肩髖膝中點夾角最大值", "站立時")
                record(cat, move, min(clean), "肩髖膝中點夾角最小值", "蹲下時")
                if (val := normalized(lk_d[min_idx], min_idx)): record(cat, move, val, "左膝到髖中點距離", "蹲下時")
                if (val := normalized(rk_d[min_idx], min_idx)): record(cat, move, val, "右膝到髖中點距離", "蹲下時")
                if (val := normalized(lk_d[max_idx], max_idx)): record(cat, move, val, "左膝到髖中點距離", "站立時")
                if (val := normalized(rk_d[max_idx], max_idx)): record(cat, move, val, "右膝到髖中點距離", "站立時")
                if nose_l[max_idx]: record("上肢", move, nose_l[max_idx], "鼻左肩左肘夾角", "肩髖膝中點夾角最大 (站立)時")
                if nose_r[max_idx]: record("上肢", move, nose_r[max_idx], "鼻右肩右肘夾角", "肩髖膝中點夾角最大 (站立)時")
                if nose_l[min_idx]: record("上肢", move, nose_l[min_idx], "鼻左肩左肘夾角", "肩髖膝中點夾角最小 (蹲下)時")
                if nose_r[min_idx]: record("上肢", move, nose_r[min_idx], "鼻右肩右肘夾角", "肩髖膝中點夾角最小 (蹲下)時")

                # DSWAO Front-Torso Tilt（左右肩髖膝）
                for side in [("L", "左"), ("R", "右")]:
                    a, b, c = f"{side[0]}Shoulder", f"{side[0]}Hip", f"{side[0]}Knee"
                    angles_side = [angle_between(pt(a, i), pt(b, i), pt(c, i)) for i in range(N)]
                    if angles_side[min_idx]:
                        record("軀幹", move, angles_side[min_idx], f"{side[1]}肩{side[1]}髖{side[1]}膝夾角", "肩髖膝中點夾角最小 (蹲下)時")

        return results

# === 主執行邏輯 ===
summary = []
for fname in os.listdir(INPUT_FOLDER):
    if fname.endswith(".csv"):
        try:
            pose = Pose3DCSV(os.path.join(INPUT_FOLDER, fname))
            summary += pose.analyze()
        except Exception as e:
            print(f"[!] Failed on {fname}: {e}")

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["Category", "Movement", "Value", "Subject", "Type", "Timing"])
    writer.writerows(summary)

print(f"✅ 分析完成，共輸出 {len(summary)} 筆資料 -> {OUTPUT_CSV}")
