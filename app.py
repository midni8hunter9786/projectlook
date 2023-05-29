from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the collection of thief images
thief_images = [cv2.imread('thief_images/thief1.jpg'), cv2.imread('thief_images/thief2.jpg')]  # List of thief images


def detect_thief():
    # Initialize the video capture
    cap = cv2.VideoCapture(0)  # Change the argument to a video file path if using a pre-recorded video

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Extract the face region from the frame
            face_roi = frame[y:y + h, x:x + w]

            # Perform template matching with thief images
            for thief_image in thief_images:
                # Resize the thief image to match the face size
                resized_thief = cv2.resize(thief_image, (w, h))

                # Perform template matching
                result = cv2.matchTemplate(face_roi, resized_thief, cv2.TM_CCOEFF_NORMED)

                # Define a threshold for template matching results
                threshold = 0.8

                # Find locations where the template matches
                locations = np.where(result >= threshold)
                for loc in zip(*locations[::-1]):
                    # Draw a rectangle around the matched area
                    cv2.rectangle(frame, (x + loc[0], y + loc[1]), (x + loc[0] + w, y + loc[1] + h),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, 'Thief', (x + loc[0], y + loc[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 255), 2)

        # Convert the frame to JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Yield the JPEG frame as a response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    # Release the capture
    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(detect_thief(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
