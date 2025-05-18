import os
import numpy as np
import cv2
import mediapipe as mp
import math
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
import tensorflow as tf

app = Flask(__name__)

# Load the pre-trained Keras classifier model (on top of features)
#model = load_model(r'model/resnet_model.h5')
model = load_model(r'C:\Users\PC\OneDrive\Desktop\final fyp\eyesync\backend\model\resnet_model.h5')

# Initialize ResNet50 base + Flatten as feature extractor (same as training)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.3)

known_interocular_distance_cm = 6.3

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def extract_features(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image_bytes = BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)

    image_array = np.frombuffer(image_bytes.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        print(f"Error loading image")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            left_eye = (int(face_landmarks.landmark[33].x * w), int(face_landmarks.landmark[33].y * h))
            right_eye = (int(face_landmarks.landmark[263].x * w), int(face_landmarks.landmark[263].y * h))
            left_brow = (int(face_landmarks.landmark[70].x * w), int(face_landmarks.landmark[70].y * h))
            right_brow = (int(face_landmarks.landmark[300].x * w), int(face_landmarks.landmark[300].y * h))
            nose_tip = (int(face_landmarks.landmark[1].x * w), int(face_landmarks.landmark[1].y * h))
            medial_canthus_left = (int(face_landmarks.landmark[133].x * w), int(face_landmarks.landmark[133].y * h))
            medial_canthus_right = (int(face_landmarks.landmark[362].x * w), int(face_landmarks.landmark[362].y * h))
            lateral_canthus_left = (int(face_landmarks.landmark[133].x * w), int(face_landmarks.landmark[133].y * h))
            lateral_canthus_right = (int(face_landmarks.landmark[362].x * w), int(face_landmarks.landmark[362].y * h))
            alar_left = (int(face_landmarks.landmark[2].x * w), int(face_landmarks.landmark[2].y * h))
            alar_right = (int(face_landmarks.landmark[98].x * w), int(face_landmarks.landmark[98].y * h))

            interocular_distance_px = calculate_distance(lateral_canthus_left, lateral_canthus_right)
            px_to_cm_ratio = known_interocular_distance_cm / interocular_distance_px

            features = {
                "brow_lid_margin_distance_left": calculate_distance(left_brow, left_eye) * px_to_cm_ratio,
                "brow_pupil_distance_left": calculate_distance(left_brow, nose_tip) * px_to_cm_ratio,
                "lid_margin_pupil_distance_left": calculate_distance(left_eye, nose_tip) * px_to_cm_ratio,
                "brow_lateral_canthal_distance_left": calculate_distance(left_brow, lateral_canthus_left) * px_to_cm_ratio,
                "canthal_nasal_alar_distance_left": calculate_distance(lateral_canthus_left, alar_left) * px_to_cm_ratio,
                "brow_alar_distance_left": calculate_distance(left_brow, alar_left) * px_to_cm_ratio,
                "brow_medial_canthal_distance_left": calculate_distance(left_brow, medial_canthus_left) * px_to_cm_ratio,
                "brow_lid_margin_distance_right": calculate_distance(right_brow, right_eye) * px_to_cm_ratio,
                "brow_pupil_distance_right": calculate_distance(right_brow, nose_tip) * px_to_cm_ratio,
                "lid_margin_pupil_distance_right": calculate_distance(right_eye, nose_tip) * px_to_cm_ratio,
                "brow_lateral_canthal_distance_right": calculate_distance(right_brow, lateral_canthus_right) * px_to_cm_ratio,
                "canthal_nasal_alar_distance_right": calculate_distance(lateral_canthus_right, alar_right) * px_to_cm_ratio,
                "brow_alar_distance_right": calculate_distance(right_brow, alar_right) * px_to_cm_ratio,
                "brow_medial_canthal_distance_right": calculate_distance(right_brow, medial_canthus_right) * px_to_cm_ratio,
            }

            return features, image  # Return both features and original image
    else:
        return None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    features = None  
    result_link = None  

    if request.method == 'POST':
        if 'image' not in request.files:
            error = "No file uploaded"
            return render_template('index.html', error=error)

        image_file = request.files['image']

        if image_file.filename == '':
            error = "No selected file"
            return render_template('index.html', error=error)

        try:
            # Ensure the result folder exists
            result_folder = os.path.join(app.root_path, 'result')
            os.makedirs(result_folder, exist_ok=True)

            # Create a secure file path
            filename = image_file.filename
            filepath = os.path.join(result_folder, filename)

            # Save the uploaded image
            image_file.save(filepath)

            # Open and process the image (for feature extraction)
            image = Image.open(filepath).convert('RGB')
            image = np.array(image)

            # Extract features
            features = extract_features(image)

            if features is None:
                error = "Could not extract features from image"
                return render_template('index.html', error=error)

            # Predict using the model
            feature_array = np.array(list(features.values())).reshape(1, -1)
            prediction = model.predict(feature_array)
            result = "Symmetric" if prediction[0] == 1 else "Asymmetric"

            # Generate result link
            result_link = url_for("result", label=result, **features)
            print(f"Generated result_link: {result_link}")

        except Exception as e:
            error = f"Error: {str(e)}"
            print(f"Exception occurred: {error}")

    return render_template('index.html', result=result, error=error, result_link=result_link)


@app.route("/result", methods=["GET"])
def result():
    label = request.args.get("label")
    features = {key: float(value) for key, value in request.args.items() if key != "label"}

    formatted_features = {}
    adjustments = {}

    for key, value in features.items():
        formatted_features[key] = f"{value:.2f} cm"

        if label == "Asymmetric":
            if "left" in key:
                adjustments[key] = f"(Adjustment: {value + 0.1:.2f} cm)"
            elif "right" in key:
                adjustments[key] = f"(Adjustment: {value - 0.1:.2f} cm)"

    return render_template(
        "result.html",
        label=label,
        features=formatted_features,
        adjustments=adjustments if label == "Asymmetric" else None
    )

if __name__ == '__main__':
    app.run(debug=True)