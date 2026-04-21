# Civic-Issue-SIH_2025
 # 🌊 AI-Based Water Logging Detection System

##  Description

The **AI-Based Water Logging Detection System** is a real-time intelligent monitoring solution designed to detect water accumulation on roads using live camera feeds. This system leverages **Computer Vision and Deep Learning** to automatically identify water presence and generate structured reports for faster decision-making.

This project is inspired by the problem statement of **Smart India Hackathon (SIH) 2025 – Civic Issue Reporting and Resolution System**, where the goal is to build smart solutions for detecting and reporting urban issues efficiently.

The system focuses on **automation, reliability, and scalability**, reducing manual monitoring efforts and enabling smart city infrastructure.

---

##  Problem Statement

Water logging is a major issue in urban areas, leading to:
- Traffic congestion   
- Road accidents   
- Poor drainage management  

Currently, detection is manual and slow.  
👉 This system automates detection and reporting in real-time.

---

##  Features

-  Real-time water detection using live camera feed  
-  AI-based classification using CNN (TensorFlow/Keras)  
-  Confidence-based detection using thresholding  
-  Stability (streak) mechanism to reduce false positives  
-  Automated reporting with image, timestamp, and location  
-  Flask backend for API and video streaming  
-  Live monitoring dashboard  

---

##  Tech Stack

- **Language:** Python  
- **Backend:** Flask  
- **Libraries:** OpenCV, NumPy, Pandas  
- **Machine Learning:** TensorFlow / Keras (CNN Model)  
- **Frontend:** HTML  

---

##  System Workflow

1. Capture live video using OpenCV  
2. Extract frames from video stream  
3. Preprocess frames (resize, RGB conversion, reshape)  
4. Pass frames to trained CNN model  
5. Predict water presence based on confidence score  
6. Apply threshold + streak logic for reliable detection  
7. Capture image on confirmed detection  
8. Store report with timestamp and location in CSV  
9. Display real-time status on dashboard  

---

##  Model Details

- Model Type: Convolutional Neural Network (CNN)  
- Input Size: 128 × 128 × 3  
- Output: Binary classification (Water / No Water)  
- Threshold: 0.92  
- Detection Logic: Temporal validation using streak counter  

---

##  Project Structure

├── static/
│ └── reports/ # Saved detection images
├── templates/
│ └── index.html # Dashboard UI
├── water_model.keras # Trained model
├── civic_reports.csv # Detection logs
├── app.py # Flask backend
└── README.md


---

##  How to Run

```bash
pip install opencv-python flask tensorflow numpy pandas
python app.py
```
Open in browser:  
http://localhost:5000

## Output
Real-time video with detection overlay
Captured image on confirmed detection
CSV report containing:
Timestamp
Location
Image path
Confidence

Limitations
Performance may vary in extreme lighting conditions
Limited dataset may affect accuracy
Currently supports only water detection


Future Scope
Extend to:
  Pothole detection
  Garbage detection
  Traffic violations
Cloud deployment for scalability
Alert system (SMS/Email)
Improve model accuracy with larger datasets

Author

Vipul Shrivastav
B.Tech CSE (Final Year)

