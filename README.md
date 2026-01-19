# YOLOv7 + OCR + TTS: Vision-to-Speech AI Microservice

This project integrates **object detection**, **text recognition (OCR)**, and **text-to-speech (TTS)** into one microservice.  
It detects visual objects using **YOLOv7**, extracts text from cropped regions, and sends it to a TTS microservice for narration — enabling assistive capabilities for visually impaired users.

---

##  Features
-  Object detection with **YOLOv7**
-  OCR integration through REST API
-  TTS integration through REST API
-  Flask-based REST API endpoints (`/upload`, `/`)
-  Dockerized and deployed on **AWS Fargate**
-  Tested in a microservice ecosystem (YOLO → OCR → TTS)

---

##  Tech Stack
| Component | Description |
|------------|-------------|
| **YOLOv7** | Object detection backbone |
| **Flask** | REST API |
| **OpenCV, NumPy** | Image processing |
| **Requests** | Inter-service communication |
| **Docker** | Packaging and deployment |
| **AWS Fargate** | Cloud hosting |

---

##  Local Setup

###  Clone repository
```bash
git clone https://github.com/<your-username>/yolov7-ocr-tts.git
cd yolov7-ocr-tts

### Install dependencies
pip install -r requirements.txt

### Run the API
python app.py

### Test upload endpoint
curl -X POST -F "image=@sample.jpg" http://localhost:5000/upload

### Docker Deployment
Build image
docker build -t yolov7-ocr-tts .

Run container
docker run -it --rm -p 5000:5000 yolov7-ocr-tts

Access thee ready built container using 

### Docker Hub Image:
docker pull kfchacha/yolo_app_version-3:latest

expose port 5000

## Author

Kenyatta Peter Chacha
Network & DevOps Engineer | AI Developer
📍 Nairobi, Kenya



