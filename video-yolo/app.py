from flask import Flask, render_template, Response
import cv2
import torch

app = Flask(__name__)


pt_file_path = "./best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', pt_file_path)       

video_path = "./web_test.mp4"
cap = cv2.VideoCapture(video_path)



@app.route('/')
def video_show():
    return render_template('index.html')

def gen_frames():
    while True:
        _, frame = cap.read()
        if not _:
            #frame = cv2.flip(frame, -1)    
            break
        else:
            #frame = cv2.resize(frame, (1280, 1080))
            results = model(frame)
            annotated_frame = results.render()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)