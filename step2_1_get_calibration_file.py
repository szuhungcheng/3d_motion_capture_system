import numpy as np
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
from aniposelib.utils import load_pose2d_fnames
import datetime

vidnames = [['output_person_000_chess_board_cam1_2025-04-24_16-27-57.mp4'],
           ['output_person_000_chess_board_cam2_2025-04-24_16-27-57.mp4'],
           ['output_person_000_chess_board_cam3_2025-04-24_16-27-57.mp4']]

cam_names = ['A', 'B', 'C']

n_cams = len(vidnames)

board = CharucoBoard(10, 7,
                     square_length=59, # here, in mm but any unit works
                    marker_length=44,
                    marker_bits=4, dict_size=50)


# the videos provided are fisheye, so we need the fisheye option
cgroup = CameraGroup.from_names(cam_names, fisheye=False)

# this will take a few minutes
# it will detect the charuco board in the videos,
# then calibrate the cameras based on the detections, using iterative bundle adjustment
cgroup.calibrate_videos(vidnames, board)

# if you need to save and load
# example saving and loading for later

cgroup.dump(f'calibration.toml')

