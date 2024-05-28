from utils.video import  SaveVideo, frame_generator
import cv2
from utils.my_tracker import Tracker
from utils.assign_teams import TeamAssigner


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video/result_output.avi', fourcc, 25, (1280, 720))

def main():

    video_path = 'video/case_study_video.mov'
    tracker = Tracker('models/football.pt')
    
    # Using the generator to get batches of frames
    frame_gen = frame_generator(video_path)

    frame_num = 0
    for bs,frame_batch in enumerate(frame_gen):    
        for img in frame_batch:
            try:
                frame,tracks = tracker.get_object_tracks(img,frame_num,
                                                    read_from_stub=False,
                                                stub_path='models/track_stubs.pkl')
                frame_num = frame_num + 1 
            except:
                pass    

            cv2.imshow('output frame',frame)

            # save frame
            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 
      
    # out.release()

if __name__=='__main__':
    main()