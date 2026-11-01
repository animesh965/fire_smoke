Fire and Smoke Detection System 🔥
About the Project

This project is about detecting fire and smoke using Python and a trained YOLO model. I built this project to explore how computer vision and deep learning can be used to identify fire and smoke from images.

The backend is developed using FastAPI. The idea is to make it possible to upload an image and get the detection results through an API.

Features
Detects fire and smoke using a trained YOLO model.
Uses FastAPI for the backend.
Supports image uploads.
Provides detection results with confidence scores.
Includes Swagger UI for testing API endpoints.
Keeps the project files organized for easier development.
Technologies Used
Python
YOLO (Ultralytics)
FastAPI
OpenCV
NumPy
Pillow
Uvicorn
Project Structure
text
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

Installation and Setup

Follow these steps to run the project on your computer.

1. Clone the Repository
bash
git clone https://github.com/animesh965/fire_smoke.git

2. Open the Project Folder
bash
cd fire_smoke

3. Create a Virtual Environment
bash
python -m venv venv

4. Activate the Environment

For Windows:

powershell
venv\Scripts\activate


For Linux or macOS:

bash
source venv/bin/activate

5. Install the Required Libraries
bash
pip install -r requirements.txt

Running the Project

To start the FastAPI server, run:

bash
uvicorn fire_backend_api:app --reload


Once the server starts, open this link in your browser:

http://127.0.0.1:8000

API Documentation

FastAPI provides an interactive page where you can check and test the available API endpoints.

http://127.0.0.1:8000/docs

Model Details

I have included my trained YOLO model in the models folder.

Model name: fire_smoke_r3_e6_final.pt

This model is intended to detect fire and smoke in images.

Project Progress
Created the project structure.
Set up the FastAPI backend.
Uploaded the project to GitHub.
Connect the trained YOLO model to the API.
Add an image upload and prediction endpoint.
Generate and save detection results.
Test the complete application.
Future Improvements

There are a few things I would like to add to this project in the future:

Real-time fire and smoke detection through video.
Webcam support.
Alerts when fire or smoke is detected.
Improvements in detection accuracy.
A simple web interface to make the project easier to use.
Disclaimer

This project is developed for learning and educational purposes. It is not meant to replace professional fire detection systems or certified fire safety equipment.

Author

Animesh

GitHub: animesh965

License

This project is licensed under the MIT License. Check the LICENSE file for more information.
