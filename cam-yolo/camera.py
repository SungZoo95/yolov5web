import torch 
import cv2
import os
import numpy as np

pt_file_path = "./best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', pt_file_path)       

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.current_frame = None

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        results = model(frame)
        a = np.squeeze(results.render())

        desired_classes = {1, 2, 3}
        detected_classes = set()
        

        for i, det in enumerate(results.pred):  # per image
            if len(det):
                for c in det[:, 5].unique():
                    detected_classes.add(int(c))
            print(detected_classes)

            if desired_classes :
                save_dir = "photos"
                file_num = len(os.listdir(save_dir))
                save_path = os.path.join(save_dir, f"detected_frame_{file_num}.jpg")
                cv2.imwrite(save_path, frame)

            break
        
        detected_classes.clear()

        # Render result
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
