 for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']
        frame_to_save = img
        cropped_image = frame_to_save[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        cv2.imwrite(f'data/players_images/player_{track_id}.jpg',cropped_image)

        break