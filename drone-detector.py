from ultralytics import YOLO

def train_model():
    model = YOLO('yolov8n.pt')
    model.train(data='drone-config.yaml', epochs=10, imgsz=640, batch=16, workers=4, name='drone_detector')

#train_model()

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from collections import deque

# --- CONFIGURATION ---
MODEL_PATH = 'runs/detect/drone_detector5/weights/best.pt'
VIDEO_DIR = 'drone_videos/'
IMG_OUTPUT_DIR = 'detections/'
VIDEO_OUTPUT_DIR = 'processed_outputs/'

# Ensure directories exist
for d in [IMG_OUTPUT_DIR, VIDEO_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# --- KALMAN TRACKER CLASS ---
class DroneTracker:
    def __init__(self, initial_bbox):
        # State: [x, y, w, h, vx, vy]
        self.kf = KalmanFilter(dim_x=6, dim_z=4)
        
        # F: State Transition Matrix (Constant Velocity Model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0], 
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0], 
            [0, 0, 0, 0, 0, 1]
        ])
        
        # H: Measurement Function
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0]
        ])

        # Initialize with the first YOLO detection
        self.kf.x[:4] = np.array(initial_bbox).reshape(4, 1)
        self.kf.P *= 10.0   # Initial Uncertainty
        self.kf.R *= 1.0    # Measurement Noise (Trust in YOLO)
        self.kf.Q *= 0.01   # Process Noise (Smoothness of flight)
        
        self.history = deque(maxlen=30) # For the 2D trajectory polyline
        self.missed_frames = 0

    def predict(self):
        self.kf.predict()
        self.missed_frames += 1
        pos = self.kf.x[:2].flatten()
        self.history.append((int(pos[0]), int(pos[1])))
        return self.kf.x[:4].flatten()

    def update(self, detection):
        self.kf.update(np.array(detection))
        self.missed_frames = 0

# --- MAIN EXECUTION ---
def run_pipeline():
    model = YOLO(MODEL_PATH)
    
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith('.mp4')]
    
    for video_name in video_files:
        print(f"--- Processing: {video_name} ---")
        cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video_name))
        
        # VideoWriter Setup (Task 2)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(VIDEO_OUTPUT_DIR, f"tracked_{video_name}")
        video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        tracker = None
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # process every 5th frame
            if frame_idx % 5 == 0:
                results = model(frame, verbose=False)[0]
                detections = [box.xywh[0].cpu().numpy() for box in results.boxes]

                # predition
                if tracker:
                    tracker.predict()

                # update tracker
                if len(detections) > 0:
                    # save raw detection frame
                    img_name = f"{video_name}_frame_{frame_idx}.jpg"
                    cv2.imwrite(os.path.join(IMG_OUTPUT_DIR, img_name), frame)
                    
                    if tracker is None:
                        tracker = DroneTracker(detections[0])
                    else:
                        tracker.update(detections[0])

                # video tracking
                if tracker and tracker.missed_frames < 15:
                    # Draw Polyline (Trajectory)
                    if len(tracker.history) > 1:
                        pts = np.array(list(tracker.history), np.int32)
                        cv2.polylines(frame, [pts], False, (0, 255, 255), 2)

                    # draw Kalman Box, with detection label
                    x, y, w, h = tracker.kf.x[:4].flatten()
                    color = (0, 255, 0) if tracker.missed_frames == 0 else (0, 165, 255)
                    label = "DETECTED" if tracker.missed_frames == 0 else "PREDICTED"
                    
                    cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), color, 2)
                    cv2.putText(frame, label, (int(x-w/2), int(y-h/2)-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # write frame to video deliverable
                    video_writer.write(frame)
                else:
                    tracker = None # reset if lost for too long

            frame_idx += 1

        cap.release()
        video_writer.release()
        print(f"Done. Video saved to: {out_path}")

if __name__ == "__main__":
    run_pipeline()