import cv2
import os
import datetime
import numpy as np
import pandas as pd
import time
from flask import Flask, render_template, Response, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- SETTINGS ---
MODEL_FILE = 'water_model.keras'
REPORT_DIR = 'static/reports'
DATABASE_FILE = 'civic_reports.csv'

THRESHOLD = 0.92       # High certainty required
REQUIRED_STREAK = 10    # 10 consecutive hits = 30 seconds of proof
INTERVAL = 3           # 3-second sampling

os.makedirs(REPORT_DIR, exist_ok=True)
MODEL = load_model(MODEL_FILE)

# Global tracking for Parameters & Stats
system_stats = {
    "current_score": 0,
    "max_confidence": 0,
    "total_reports": 0,
    "streak": 0,
    "status": "System Idle"
}

current_location = "Not Set"
monitoring_active = False
streak_counter = 0 

def gen_frames():
    global current_location, monitoring_active, streak_counter
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success: break

        if monitoring_active:
            # 1. AI Pre-processing
            img_resized = cv2.resize(frame, (128, 128))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_array = np.expand_dims(img_rgb, axis=0)
            
            # 2. Prediction
            prediction = MODEL.predict(img_array, verbose=0)[0][0]
            score = float(prediction)
            
            # Update Stats for Dashboard
            system_stats["current_score"] = round(score * 100, 2)
            if score * 100 > system_stats["max_confidence"]:
                system_stats["max_confidence"] = round(score * 100, 2)
            
            # 3. Temporal Logic (Streak)
            if score > THRESHOLD:
                streak_counter += 1
                system_stats["status"] = "Sustained Water Detected"
            else:
                streak_counter = 0 
                system_stats["status"] = "Clear / Obstruction"

            system_stats["streak"] = streak_counter

            # 4. Report Trigger
            if streak_counter >= REQUIRED_STREAK:
                now = datetime.datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(REPORT_DIR, f"flood_{timestamp}.jpg")
                
                cv2.imwrite(file_path, frame)
                
                log_entry = {
                    "Timestamp": [now.strftime("%Y-%m-%d %H:%M:%S")],
                    "Location": [current_location],
                    "Image_Path": [file_path],
                    "Confidence": [f"{score*100:.2f}%"]
                }
                pd.DataFrame(log_entry).to_csv(DATABASE_FILE, mode='a', 
                            header=not os.path.exists(DATABASE_FILE), index=False)
                
                system_stats["total_reports"] += 1
                streak_counter = 0 # Reset after filing
                print(f"REPORT FILED: {file_path}")

            # 5. Visual Overlays on Video Feed
            color = (0, 0, 255) if streak_counter > 0 else (0, 255, 0)
            cv2.putText(frame, f"STREAK: {streak_counter}/{REQUIRED_STREAK}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (10, 60), (10 + (streak_counter * 20), 75), (0, 255, 255), -1)

            time.sleep(INTERVAL)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_stats')
def get_stats():
    return jsonify(system_stats)

@app.route('/start_system', methods=['POST'])
def start_system():
    global current_location, monitoring_active, streak_counter
    current_location = request.json.get('location', 'Unknown Kanpur Area')
    streak_counter = 0
    monitoring_active = True
    return jsonify({"status": "active", "location": current_location})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, port=5000)
