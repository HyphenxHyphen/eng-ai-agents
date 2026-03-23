Video 1: https://youtu.be/_kfBWOOzBWs


Video 2: https://youtu.be/_JZJghX3_fo


HuggingFace Dataset: https://huggingface.co/datasets/HyphenxHyphen/DroneDetectionDataset6613JL/blob/main/drone_detections.parquet


Dataset choice and detector configuration.


I chose to use a trimmed version of the Seraphim Drone Detection Dataset (https://huggingface.co/datasets/lgrzybowski/seraphim-drone-detection-dataset), which was a massive dataset of over 80k images. At the start, I thought that the more images I trained on, the better the detection. However, as the training progressed I realized that a single epoch took over 1.5 hours to complete, and it was unfeasible to train for 25 epochs. Instead, I trimmed the training dataset down to the first 10k images, and lowered the epochs to 10. Even then, it took too long. I noticed that mAP50 was ~0.86 at the 5th-6th epoch, with barely any improvements between epoch 5 and 6, so I manually stopped training at the start of the 7th epoch. The Seraphim Dataset features images taken of drones, not from drones, is a curated collection of other sources, and as required, configured for YOLO. The Ultralytics YOLOv8n (Nano) model was chosen for its lightweight speed, I'm using a laptop GPU (NVIDIA RTX 2060). Input size was 640x640 pixels, matching video resolution. SGD was set with a learning rate of 0.01.


Kalman filter state design and noise parameters.


The tracker uses a Constant Velocity (CV) motion model. The state vector x is defined as x = [u, v, w, h, ~u, ~v]^T, Where (u, v) is the center of the drone's bounding box, (w, h) are the dimensions, and (~u, ~v) represent the horizontal and vertical velocity (estimate). Estimating velocity allows the filter to "reason" (predict?) where the drone is even when the detector fails for a few frames. Probabilistic reasoning is controlled by two main covariance matrices: Process Noise and Measurement Noise. Process Noise (Q) is set to a low value (0.01), which assumpts that a drone's flight path is relatively smooth and follows regular physics between frames. Measurement Noise (R) is set to 1.0, which tells the system that while YOLOv8 is accurate, its bounding boxes may "jitter" slightly. The Kalman Filter uses this to smooth the 2D trajectory rather than following every minor detection jump.


Failure cases and how the tracker handles missed detections.


The Prediction step handles missed detections. If the model fails to detect a drone for a few frames (less than 10), the tracker enters Prediction-only, which attempts to predict the drone's location using previously known information (velocity). The Kalman filter adjusts the bounding box (forward) based on that information. Once the drone is detected again, the Kalman Gain calculates the error between where it guessed the drone is and where the drone actually is and updates the drone's position and velocity.

However, if a target is lost for more than 10 frames (2 seconds), perhaps due to the drone moving out of frame, or performing a rapid aerial movement, or moves behind an object for a long time, the tracker will stop tracking entirely.