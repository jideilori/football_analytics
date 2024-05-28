from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
from utils.bbox import get_center_of_bbox, get_bbox_width, get_foot_position
from utils.assign_teams import TeamAssigner
from utils.df_write import create_and_write_csv


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.team_assigner = TeamAssigner()
        self.init_team_assigner = 0
        self.add_row_to_csv = create_and_write_csv('result_data.csv')
        self.color_name = None

    def detect_frames(self, frames):
        return self.model.predict(frames,conf=0.08,verbose=False)[0]
    

    def get_object_tracks(self, frames,frame_count,read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        yolo_results = self.detect_frames(frames)

        detection_supervision = sv.Detections.from_ultralytics(yolo_results)


        # To store tracking data
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        frame_num = 0
        for _ , detection in enumerate(yolo_results):

            cls_names = detection.names   
            cls_names_inv = {v:k for k,v in cls_names.items()}
           
            
            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
          

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
          
            for dc,frame_detection in enumerate(detection_with_tracks):
                # print('dc===',dc,frame_detection)
                # dc=== 0 (array([     538.02,      204.17,      555.24,      255.63], dtype=float32), None, 0.91566014, 3, 1, {'class_name': 'referee'})
                # dc=== 1 (array([     864.96,       431.4,      895.08,       528.8], dtype=float32), None, 0.9067143, 2, 2, {'class_name': 'player'})

                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]


                if cls_id == cls_names_inv['player']:
                    # Key- trackid : val - bbox

                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}

                    # if cls_id == cls_names_inv['ball']:
                    #     tracks["ball"][frame_num][track_id] = {"bbox":bbox}
                       
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

            img = frames.copy()
            if self.init_team_assigner==0:
                self.team_assigner.assign_team_color(img, tracks['players'][0])
            self.init_team_assigner=1
            
            for fn, player_track in enumerate(tracks['players']):

                for player_id, track in player_track.items():
                    team = self.team_assigner.get_player_team(img,   
                                                        track['bbox'],
                                                        player_id)
                    tracks['players'][fn][player_id]['team'] = team 
                    tracks['players'][fn][player_id]['team_color'] = self.team_assigner.team_colors[team]


            # ---------------------tracker-----------------------------------
            track_frame = img.copy()
            

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                # print('player=========',player)
                color = player.get("team_color",(0,0,255))
                track_frame = self.draw_track_ellipse(track_frame, player["bbox"],color, track_id)
                


                # -----------------------LOG DATA START---------------------------
                cx,cy= get_foot_position(player["bbox"])
                color_array = list(color)
                if color_array==[255.0, 0.0, 0.0]:
                    self.color_name ='red'
                else:
                    self.color_name='blue'
                
                self.add_row_to_csv(frame_count,track_id,cx,cy,self.color_name)

                # -----------------------LOG DATA END---------------------------


                if player.get('has_ball',False):
                    track_frame = self.draw_traingle(track_frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                track_frame = self.draw_track_ellipse(track_frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                track_frame = self.draw_traingle(track_frame, ball["bbox"],(0,255,0))


            frame_num = frame_num + 1

            if stub_path is not None:
                with open(stub_path,'wb') as f:
                    pickle.dump(tracks,f)

            return track_frame,tracks


    def draw_track_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        
        
        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
           
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
     
        return frame

           
    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame



    def draw_track_annotations(self,video_frames,frame_num, tracks):
        # output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            # frame_num = frame_num_count + frame_num
            # frame_num = 0

            frame = video_frames.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_track_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_track_ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

            return frame
            # output_video_frames.append(frame)
            # print(frame_num)
            # cv2.imshow('track frame',frame)
            # cv2.waitKey(0)
        # print('output frames ---',len(output_video_frames))
        # return output_video_frames