Fire and Smoke Detection System

-> Project Overview

The Fire and Smoke Detection System is a Python-based computer vision project designed to detect fire and smoke using a trained YOLO deep learning model.

The system provides a backend API built with FastAPI, allowing users to upload images and receive fire and smoke detection results.

The main objective of this project is to support early fire and smoke identification through automated image analysis.


-> Features

Fire detection using a trained YOLO model.
Smoke detection using computer vision.
FastAPI-based backend.
Image upload support.
Detection results with confidence scores.
REST API with interactive Swagger documentation.
Organized project structure.

-> Technologies Used
Python
YOLO (Ultralytics)
FastAPI
OpenCV
NumPy
Pillow
Uvicorn


-> Project Structure


fire-smoke-detection/
│
├── fire_backend_api.py
│
├── app/
│   ├── __init__.py
│   └── config.py
│
├── models/
│   ├── README.md
│   └── fire_smoke_r3_e6_final.pt
│
├── fire_smoke_test_results/
│   └── .gitkeep
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── LICENSE

-> Installation and Setup
1. Clone the Repository
bash
git clone https://github.com/animesh965/fire_smoke.git

2. Navigate to Project Directory
bash
cd fire_smoke

3. Create a Virtual Environment
bash
python -m venv venv

4. Activate Virtual Environment

Windows:

powershell
venv\Scripts\activate


Linux / macOS:

bash
source venv/bin/activate

5. Install Dependencies
bash
pip install -r requirements.txt

=> Run the Application

Start the FastAPI server:

bash
uvicorn fire_backend_api:app --reload


The application will run at:

text
http://127.0.0.1:8000

API Documentation

Open the Swagger UI in your browser:

text
http://127.0.0.1:8000/docs

-> Model Information

The project uses a trained YOLO model:

Model File: fire_smoke_r3_e6_final.pt

The model is intended for identifying fire and smoke in images.

-> Project Status
Project structure created.
Backend API initialized.
GitHub repository created and code uploaded.
YOLO model integration.
Image upload and prediction endpoint.
Detection result generation.
Complete system testing.


-> Future Improvements
Real-time fire and smoke detection using video streams.
Webcam integration.
Detection alerts and notifications.
Improved model accuracy.
Web-based user interface.

! Disclaimer

This project is intended for educational and research purposes. It should not be considered a replacement for certified fire alarm systems or professional fire safety equipment.

-> Author

Animesh

GitHub: https://github.com/animesh965

-> License

This project is licensed under the MIT License. See the LICENSE file for details.
