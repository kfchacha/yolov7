import os
import shutil
import base64
import cv2
import requests
import numpy as np
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify
import requests  # Make sure you have the requests library installed
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)

# Define paths for saving images and outputs
TEMP_OUTPUT_PATH = '/Yolo_Object_Detection/venv/detections_try'
FINAL_OUTPUT_PATH = '/Yolo_Object_Detection/venv/detections_try'
os.makedirs(TEMP_OUTPUT_PATH, exist_ok=True)
os.makedirs(FINAL_OUTPUT_PATH, exist_ok=True)

# Run YOLOv7 detection
def run_detection(image_path):
    subprocess.run([
        'python', 'detect.py',
        '--weights', '/Yolo_Object_Detection/venv/runs/train/exp22/weights/best.pt',
        '--conf', '0.1',
        '--source', image_path,
        '--save-txt',
        '--project', TEMP_OUTPUT_PATH,
        '--name', '.',
        '--exist-ok'
    ])

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    detected_image_path = os.path.join(TEMP_OUTPUT_PATH, base_name + '.jpg')
    final_detected_image_path = os.path.join(FINAL_OUTPUT_PATH, base_name + '.jpg')

    if os.path.exists(detected_image_path):
        shutil.move(detected_image_path, final_detected_image_path)

    # Correct label path structure
    label_path = os.path.join(TEMP_OUTPUT_PATH, 'labels', base_name + '.txt')
    final_label_path = os.path.join(FINAL_OUTPUT_PATH, base_name + '.txt')

    if os.path.exists(label_path):
        shutil.move(label_path, final_label_path)

    print(f"Detection complete. Results saved in '{FINAL_OUTPUT_PATH}'")
    return final_detected_image_path, final_label_path

# Merge and extract bounding boxes
def merge_and_crop(image_path):
    def calculate_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        return interArea / float(unionArea) if unionArea > 0 else 0

    def should_merge(boxA, boxB, iou_thresh=0.5, proximity_thresh=20):
        if calculate_iou(boxA, boxB) > iou_thresh:
            return True
        xa1, ya1, xa2, ya2 = boxA
        xb1, yb1, xb2, yb2 = boxB
        if abs(xa1 - xb2) < proximity_thresh or abs(xb1 - xa2) < proximity_thresh:
            if abs(ya1 - yb1) < proximity_thresh or abs(ya2 - yb2) < proximity_thresh:
                return True
        return False

    def merge_boxes(boxes):
        merged = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            boxA = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                boxB = boxes[j]
                if should_merge(boxA, boxB):
                    x1 = min(boxA[0], boxB[0])
                    y1 = min(boxA[1], boxB[1])
                    x2 = max(boxA[2], boxB[2])
                    y2 = max(boxA[3], boxB[3])
                    boxA = [x1, y1, x2, y2]
                    used[j] = True
            merged.append(boxA)
            used[i] = True
        return merged

    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    base_name = Path(image_path).stem
    label_path = f'{FINAL_OUTPUT_PATH}/{base_name}.txt'

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        boxes = []
        for line in lines:
            cls, x_c, y_c, w, h = map(float, line.strip().split())
            x1 = int((x_c - w / 2) * img_w)
            y1 = int((y_c - h / 2) * img_h)
            x2 = int((x_c + w / 2) * img_w)
            y2 = int((y_c + h / 2) * img_h)
            boxes.append([x1, y1, x2, y2])

        merged_boxes = merge_boxes(boxes)
        crop_output = f'{FINAL_OUTPUT_PATH}/merged_crops'
        os.makedirs(crop_output, exist_ok=True)

        crop_paths = []  # Store crop file paths to send later
        for i, (x1, y1, x2, y2) in enumerate(merged_boxes):
            roi = img[y1:y2, x1:x2]
            crop_path = os.path.join(crop_output, f'{base_name}_crop_{i}.png')
            cv2.imwrite(crop_path, roi)
            print(f"[✔] Saved merged crop: {crop_path}")
            crop_paths.append(crop_path)

        # Now send crops to API
        send_crops_to_api(crop_paths)
    else:
        print(f"[!] No label file found: {label_path}")


# Function to send the merged crops to another API
def send_crops_to_api(crop_paths):
    url = "http://Loadbalancer-version2-1665063054.eu-north-1.elb.amazonaws.com/ocr/ssml"  # Change this to OCR API's URL
    files = []

    try:
        # Open all files and keep them open during the request
        open_files = [open(path, 'rb') for path in crop_paths]
        files = [
            ('file', (os.path.basename(f.name), f, 'image/png'))
            for f in open_files
        ]

        # Send the request with the open file objects
        response = requests.post(url, files=files)

        # Close all files after the request
        for f in open_files:
            f.close()

        if response.status_code == 200:
            print(f"Successfully sent crops to API: {response.json()}")
        else:
            print(f"Failed to send crops. Status code: {response.status_code}")

    except Exception as e:
        print(f"Error sending crops to API: {str(e)}")
        # Make sure we close any opened files in case of exception
        for f in files:
            try:
                f[1][1].close()
            except:
                pass

#API for health check
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200


# API to receive image via POST request
@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    if file:
        # Save image to a temporary location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = f"/tmp/{timestamp}.jpg"
        file.save(image_path)

        # Run YOLO detection on the uploaded image
        detected_image_path, label_path = run_detection(image_path)

        # Merge and crop bounding boxes
        merge_and_crop(detected_image_path)

        # Send response back
        return jsonify({
            'status': 'success',
            'message': 'Detection complete.',
            'detected_image': detected_image_path,
            'label_file': label_path
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'No image uploaded.'
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

