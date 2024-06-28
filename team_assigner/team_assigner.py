from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        # To maintain what team a player is in
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        # Reshaping image
        image_2d = image.reshape(-1, 3)

        # Perform clustering
        # k-means++ to reduce the time taken for clusterting 
        # n_init = number of iterations
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=2)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = [int(cord) for cord in bbox]
        image = frame[y1:y2, x1:x2]
        top_image_half = image[0:int(image.shape[0]/2), :]

        # Clustering
        kmeans = self.get_clustering_model(top_image_half)

        # Get the cluster labels
        labels = kmeans.labels_

        # Reshape to image shape
        clustered_image = labels.reshape(top_image_half.shape[0], top_image_half.shape[1])

        # player cluster
        corner_clusters = [clustered_image[0,0],
                           clustered_image[0, -1],
                           clustered_image[-1, 0],
                           clustered_image[-1, -1]]

        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # get player color
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_color_to_team(self, frame, player_detections):
        
        player_colors = []

        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)

            # to get all player jersey colors
            player_colors.append(player_color)
        
        # divide the player colours into 2
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1

        goal_keeper_id = [99, 104, 128]
        if player_id in goal_keeper_id:
            team_id = 2

        self.player_team_dict[player_id] = team_id

        return team_id