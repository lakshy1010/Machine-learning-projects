# AI Sit-Up Counter Using Computer Vision

A real-time AI-based sit-up counter that uses **MediaPipe Pose Estimation** and **OpenCV** to detect body movements and automatically count repetitions using joint angle calculations.

## Features

* Real-time pose detection using webcam
* Automatic sit-up repetition counting
* Knee angle calculation for movement analysis
* Live display of pose landmarks, angle, stage, and rep count
* Works with different webcam indexes

## Tech Stack

* **Python**
* **OpenCV** – Video processing
* **MediaPipe Pose Landmarker** – Human pose detection
* **NumPy** – Mathematical calculations

## How It Works

1. Captures live video through webcam using OpenCV.
2. Detects body landmarks using MediaPipe Pose Estimation.
3. Extracts hip, knee, and ankle coordinates.
4. Calculates the knee joint angle.
5. Tracks movement stages:

   * Bent position → angle < 110°
   * Straight position → angle > 150°
6. Counts one repetition after a complete movement cycle.

##
