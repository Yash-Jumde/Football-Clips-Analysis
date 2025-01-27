import sys

sys.path.append("../")
from utils import get_centre_of_bbox, measure_distance

class PlayerBallAssigner:
    def __init__(self):
        self.max_player_and_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_centre_of_bbox(ball_bbox)

        minimun_distance = 999999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player["bbox"]

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            # get closest player
            if distance < self.max_player_and_ball_distance:
                minimun_distance = distance
                assigned_player = player_id

        return assigned_player
