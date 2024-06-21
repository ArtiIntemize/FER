from flask import Flask, render_template, request
import os
import cv2
from fer import FER
import time
from collections import defaultdict

app = Flask(__name__)

# Path to the folder containing video files
input_video_folder = 'videos'
output_video_folder = 'processed_videos'

# Ensure the folders exist
if not os.path.exists(input_video_folder):
    os.makedirs(input_video_folder)
if not os.path.exists(output_video_folder):
    os.makedirs(output_video_folder)

# Initialize the facial expression recognizer
detector = FER()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    video_file = request.files['video']
    video_filename = video_file.filename
    input_video_path = os.path.join(input_video_folder, video_filename)
    output_video_path = os.path.join(output_video_folder, f"processed_{video_filename}")
    video_file.save(input_video_path)

    # Get the frame rate and size of the video
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps
    interval_seconds = 2  # Interval in seconds to capture frames
    frame_interval = int(fps * interval_seconds)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # Initialize lists and dictionaries to store data
    expression_data = []
    expression_counts = defaultdict(int)

    # Process the video frame by frame
    frame_count = 0
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # Capture frames at the specified interval
        if frame_count % frame_interval == 0:
            # Detect emotions in the frame
            result = detector.detect_emotions(frame)

            # Process each detected face
            for person in result:
                emotions = person['emotions']
                top_emotion = max(emotions, key=emotions.get)
                confidence = emotions[top_emotion]

                # Append emotion data to the list
                expression_data.append({
                    'Emotion': top_emotion,
                    'Confidence': confidence,
                    'Capture Time': time.strftime("%H:%M:%S", time.gmtime(frame_count / fps))
                })

                # Increment count for the detected emotion
                expression_counts[top_emotion] += 1

                # Draw rectangle around the face
                x, y, w, h = person['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Put the emotion text above the rectangle
                text = f'{top_emotion} ({confidence:.2f})'
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Write the processed frame to the output video
            out.write(frame)

        # Increment frame count
        frame_count += 1

        # Introduce a delay to match the frame duration
        time_elapsed = time.time() - start_time
        if time_elapsed < frame_duration:
            time.sleep(frame_duration - time_elapsed)

    # Release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Render the template with expression data and counts
    return render_template('result.html', expression_data=expression_data, expression_counts=expression_counts)

if __name__ == '__main__':
    app.run(debug=True)

