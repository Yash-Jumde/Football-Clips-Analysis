from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assignment import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedDistanceEstimator

def main():
    # print("Hello World!")
    # Read video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # initializing tracker class
    tracker = Tracker("models/best_v5.pt")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
        )
    
    # Get positions
    tracker.add_position_to_tracks(tracks)

    # Estimation of camera movement
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path="stubs/camera_movement_stub.pkl")
    camera_movement_estimator.adjust_positions(tracks, camera_movement_per_frame)

    # Perspective transformation
    view_transform = ViewTransformer()
    view_transform.add_transformed_position(tracks)


    # interpolating ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Saving cropped image of a player to identify jersey color
    # Only run once to get 1 image to be test on in the color_assignment.ipynb
    # for track_id, player in tracks['players'][0].items():
        # bbox = player["bbox"]
        # frame = video_frames[0]

        # crop bbox from the frame
        # print(bbox)
        # bbox = [int(cord) for cord in bbox]
        # x1, y1, x2, y2 = bbox
        # cropped_image = frame[y1:y2, x1:x2]

        # save image
        # img_path = "output_videos/cropped_image.jpg"
        # cv2.imwrite(img_path, cropped_image)
        
        # We only need one image for analysis
        # break

    # Speed and distance estimator
    speed_distance_estimator = SpeedDistanceEstimator()
    speed_distance_estimator.add_speed_distance_to_tracks(tracks)
    
    # Assign teams
    team_assigner = TeamAssigner()
    team_assigner.assign_color_to_team(video_frames[0], tracks['players'][0])

    # loop over each player for each frame to assin the correct jersey color
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            # Adding new key and value pairs to "tracks"
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]


    # Assign the ball to player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            # For possesion stats
            team_ball_control.append(tracks["players"][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    
    team_ball_control = np.array(team_ball_control)
    

    # Process output video
    # draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Draw the camera movement over the frames
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,
                                                                         camera_movement_per_frame)

    # Add speed and distance to frame
    speed_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save the video
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == '__main__':
    main()