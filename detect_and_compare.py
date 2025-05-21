import cv2
import os
import sqlite3
from flask import Flask, jsonify
import pyttsx3
import threading
import mediapipe as mp
from deepface import DeepFace
from ultralytics import YOLO  # For yolo implementation

DATABASE = 'attendance.db'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/faces'

# Load YOLO model once globally
yolo_model = YOLO('yolov8n.pt')  # Use yolov8s.pt or custom trained model for better accuracy

#  YOLO-based environment check function (with detailed object reporting)
def is_environment_suitable(image):
    """Check if exactly one object is present, with detailed detection report."""
    results = yolo_model(image)
    detections = results[0].boxes.cls.tolist()

    # Count all objects by class
    from collections import Counter
    detected_objects = Counter([yolo_model.names[int(cls)] for cls in detections])

    total_objects = sum(detected_objects.values())

    if total_objects >4:
        # Generate detailed detection message
        details = ", ".join([f"{count} {name}{'s' if count >4 else ''}"
                             for name, count in detected_objects.items()])
        return False, (
            f"This Environment detected  {total_objects} objects: {details}. "
            "Let me check your face!"
        )

    # Get the single detected object's name
    object_name = next(iter(detected_objects.keys()))
    return True, f"Environment valid (1 {object_name} detected)"

def capture_and_compare_from_saved_image(image_path):
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({'status': 'error', 'message': 'Failed to load image'})

        #  Check environment with YOLO before proceeding
        suitable, env_msg = is_environment_suitable(image)
        if not suitable:
            return jsonify({'status': 'error', 'message': env_msg})

        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5)

        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Face Mesh
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return jsonify(
                {'status': 'error', 'message': 'No face detected. Please position your face clearly in the frame.'})

        # Draw face landmarks and connections
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)

            ih, iw, _ = image.shape
            landmarks = face_landmarks.landmark
            chin = landmarks[152]
            chin_x, chin_y = int(chin.x * iw), int(chin.y * ih)
            neck_length = int(0.2 * ih)
            cv2.line(image, (chin_x, chin_y), (chin_x, chin_y + neck_length), (0, 255, 0), 2)
            shoulder_width = int(0.4 * iw)
            cv2.line(image,
                     (chin_x - shoulder_width // 2, chin_y + neck_length),
                     (chin_x + shoulder_width // 2, chin_y + neck_length),
                     (0, 255, 0), 2)

        marked_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'marked.jpg')
        cv2.imwrite(marked_image_path, image)

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute("SELECT id, name, photo_path FROM users")
        users = cursor.fetchall()

        for user_id, user_name, photo_path in users:
            if not os.path.isfile(photo_path):
                print(f"⚠️ Skipping {user_name}: photo not found at {photo_path}")
                continue
            try:
                result = DeepFace.verify(
                    img1_path=image_path,
                    img2_path=photo_path,
                    model_name='VGG-Face',
                    enforce_detection=False
                )
                if result['verified']:
                    cursor.execute("INSERT INTO attendance (user_id) VALUES (?)", (user_id,))
                    conn.commit()
                    conn.close()

                    message = f'Welcome {user_name}. Attendance is marked for you.'

                    def speak(text):
                        local_engine = pyttsx3.init()
                        local_engine.say(text)
                        local_engine.runAndWait()

                    threading.Thread(target=speak, args=(message,)).start()

                    return jsonify({'status': 'success', 'message': message})
            except Exception as e:
                print(f"Error comparing with {user_name}: {e}")
                continue

        conn.close()
        message = 'Face not recognized, please position your face to the camera'

        def speak(text):
            local_engine = pyttsx3.init()
            local_engine.say(text)
            local_engine.runAndWait()

        threading.Thread(target=speak, args=(message,)).start()

        return jsonify({'status': 'error', 'message': message})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
