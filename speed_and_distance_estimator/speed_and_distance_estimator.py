import cv2

import sys
sys.path.append("../")
from utils import measure_distance, get_foot_position

class SpeedDistanceEstimator:
    def __init__(self):
        # speed of the players to be carried out in this frame window
        self.frame_window = 5
        # Frame rate of the video
        self.frame_rate = 24

    def add_speed_distance_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            if object == "ball" or object == "referees":
                continue
            number_frames = len(object_tracks)
            for frame_num in range(0, number_frames, self.frame_window):
                last_frame = min(frame_num + self.frame_window, number_frames-1)

                for track_id, _ in object_tracks[frame_num].items():
                    # If a player is in the initial frame but not the last frame of the frame window, do not calculate
                    if track_id not in object_tracks[last_frame]:
                        continue

                    start_position = object_tracks[frame_num][track_id]["transformed_position"]
                    end_position = object_tracks[last_frame][track_id]["transformed_position"]

                    # If the player is not inside the transformed perspective do not calculate
                    if start_position is None or end_position is None:
                        continue

                    covered_distance = measure_distance(start_position, end_position)
                    time_elapsed = (last_frame - frame_num)/self.frame_rate

                    # Speed in meters per second --> kilometer per second
                    speed_mps = covered_distance / time_elapsed
                    speed = speed_mps * 3.6

                    if object not in total_distance:
                        total_distance[object] = {}

                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += covered_distance

                    # annotations
                    for frame_num_in_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_in_batch]:
                            continue
                        
                        tracks[object][frame_num_in_batch][track_id]['speed'] = speed
                        tracks[object][frame_num_in_batch][track_id]['distance'] = total_distance[object][track_id]


    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball":
                    continue
                if object == "referees":
                    for _, track_info in object_tracks[frame_num].items():
                        bbox = track_info["bbox"]
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 20
                        position = tuple(map(int, position))

                        cv2.putText(frame,
                                    f"Referee",
                                    (position[0]-30, position[1]),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.5,
                                    (0,0,0),
                                    1)
                for _, track_info in object_tracks[frame_num].items():
                    if "speed" in track_info:
                        speed = track_info.get("speed", None)
                        distance = track_info.get("distance", None)
                        if speed is None or distance is None:
                            continue

                        bbox = track_info["bbox"]
                        position = get_foot_position(bbox)
                        # Adding a buffer of 40 pixels at the bottom
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(frame,
                                    f"{speed:.2f} km/hr",
                                    (position[0]-30, position[1]),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.5,
                                    (0,0,0),
                                    1)
                        cv2.putText(frame,
                                    f"{distance:.2f} m",
                                    (position[0]-30, position[1]+20),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.5,
                                    (0,0,0),
                                    1)
            output_frames.append(frame)

        return output_frames


