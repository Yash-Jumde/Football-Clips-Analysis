import pickle
import cv2
import numpy as np
import os

import sys
sys.path.append("../")
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize = (15, 15), #size of the window to be searched
            maxLevel = 2, # downscale the image to get larger features
            # Stopping criteria --> anything 10 times above 0.03 
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # top and bottom areas that will be used to extract features.
        mask_features = np.zeros_like(first_frame)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners = 100, #maximum number of corners for the good features
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7, #search size for the features
            mask = mask_features
        )

    def adjust_positions(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    adjusted_position = (position[0]-camera_movement[0], position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]["adjusted_position"] = adjusted_position


    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read Stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)

        previous_grayscale = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_features = cv2.goodFeaturesToTrack(previous_grayscale, **self.features)

        for frame_num in range(1, len(frames)):
            frame_grayscale = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            # extracting new features using optical flow
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(previous_grayscale, frame_grayscale, prev_features, None, **self.lk_params)

            # max distance between any two features
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (prev_feature, new_feature) in enumerate(zip(prev_features, new_features)):
                new_features_point = new_feature.ravel()
                prev_features_point = prev_feature.ravel()

                distance = measure_distance(new_features_point, prev_features_point)
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(prev_features_point, new_features_point)
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                prev_features = cv2.goodFeaturesToTrack(frame_grayscale, **self.features)
            
            previous_grayscale = frame_grayscale.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                # wb = write bites/bytes
                pickle.dump(camera_movement, f)

        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,
                          (20,970),
                          (550,1090),
                          (255, 255, 255),
                          -1)
            alpha=1
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,
                                f"Camera Movement X: {x_movement:.2f}",
                                (40, 1020),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1,
                                (0, 0, 0),
                                2)
            frame = cv2.putText(frame,
                                f"Camera Movement Y: {y_movement:.2f}",
                                (40, 1060),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1,
                                (0, 0, 0),
                                2)
            
            output_frames.append(frame)
        
        return output_frames



