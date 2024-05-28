from sklearn.cluster import KMeans
import cv2
import numpy as np



class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    
    def classify_teams(self,player, color): 
        '''
        this functions is to classify players to teams based on thier colors.

        Parameters
        ----------
        player : image of the player
            the dataframe of the objects.
        colors : list of strings         
            list of the colors to choose the player color from.   
            the colors available are : [white, black, red, blue, green, yellow, purple, skyblue] 
        
        Return
        ----------
        player choosen color: string      
        '''
        
        player = cv2.cvtColor(player, cv2.COLOR_BGR2HSV)
        masks = []  
        Hsv_boundaries = [
            ([0,0,185],[180,80,255], 'white'), #white
            ([0,0,0],[180,255,65], 'black'), #black
            ([0,120,20],[15,255,255], 'red'), #red
            ([163,100,20],[180,255,255], 'red'), #dark red/ pink
            ([90,100,20],[130,255,255], 'blue'), #blue
            ([50,100,20],[80,255,255], 'green'), #green
            ([17,150,20],[35,255,255], 'yellow'), #yellow       
            ([131,60,20],[162,255,255], 'purple'), #purple          
            ([80,36,20],[105,255,255], 'skyblue'), #skyblue   
            ]  
        
        filtered_boundaries = list(filter(lambda boundary: boundary[2] in color, Hsv_boundaries))  
        for boundary in filtered_boundaries:
            mask = cv2.inRange(player, np.array(boundary[0]) , np.array(boundary[1]))
            count = np.count_nonzero(cv2.bitwise_and(player, player, mask = mask))
            masks.append(count)  
        
        color_idx = masks.index(max(masks)) 

        return filtered_boundaries[color_idx][2]
        
    def get_clustering_model(self,image):
        """
        Applies K-means clustering to the given image, clustering its pixels into two groups.

        Parameters:
        image (numpy.ndarray): A 3D NumPy array representing the image with shape 
                               (height, width, channels), where channels is typically 3 for an RGB image.

        Returns:
        KMeans: A trained K-means model from scikit-learn, fitted to the 2D reshaped image data.
        """
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        """
        Determines the predominant color of a player within a specified bounding box in a video frame.

        Parameters:
        frame (numpy.ndarray): 
        bbox (tuple): A tuple of four integers representing the bounding box 
                      (x_min, y_min, x_max, y_max) defining the area containing the player.

        Returns:
        numpy.ndarray: A 1D array representing the RGB color of the player.
        """
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]        

        top_half_image = image[0:int(image.shape[0]/2),:]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame, player_detections):
        """
        Assigns team colors to players detected in a video frame.

        Parameters:
        frame (numpy.ndarray):

        player_detections (dict): A dictionary containing player detection data. Each key is a player ID, and 
                                  each value is a dictionary with a "bbox" key containing the bounding box 
                                  (x_min, y_min, x_max, y_max) of the detected player.

        Returns:
        None
        """
        
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]

            player = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

            player_color = self.classify_teams(player,['red','blue'])
            if player_color=='red':
                player_color = np.array([255, 0, 0])
            else:
                player_color = np.array([0, 0, 255])

            player_colors.append(player_color)
        
        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        """
        Determines the team of a player based on their detected color within a specified bounding box in a video frame.

        Parameters:
        frame (numpy.ndarray):.
        player_bbox (tuple): (x_min, y_min, x_max, y_max) 
        player_id (int): A unique identifier for the player.

        Returns:
        int: The team ID assigned to the player.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        self.player_team_dict[player_id] = team_id

        return team_id