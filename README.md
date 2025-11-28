# Wearable Multi-Signal Fusion & Time-Series Adaptive PTT Calibration for Continuous Blood Pressure Estimation

## Overview
This project implements a **wearable, multi-modal biosignal fusion algorithm** for highly accurate **Pulse Transit Time (PTT)**–based blood pressure (BP) estimation.

Traditional PTT–BP models suffer from:
- Individual variability  
- Sensor-position drift  
- Physiological fluctuations  
- Long-term monitoring instability

To address these issues, the system introduces a **time-series adaptive calibration framework**, enabling continuous, precise, and personalized blood pressure estimation using ECG, PPG, and IMU signals.

---

## Key Features

### 1. Multi-Modal Signal Fusion
- **ECG**: R-peak detection  
- **PPG**: foot detection and morphology analysis  
- **IMU**: motion artifact detection & SQI weighting  
- Motion-robust PTT estimation using fused signals

---

### 2. Time-Series Adaptive Calibration
- Sliding-window calibration over continuous data streams  
- Calibration function updated dynamically  
- Initial reference from cuff-based SBP/DBP measurements  
- Personalized PTT–BP curve refinement over time  

---

### 3. Dynamic Correction Function
Supports various adaptive correction models:
- Linear / polynomial regression  
- Nonlinear mapping functions  
- Kalman Filter–based dynamic updates  
- Robust against drift and long-term physiological variation

---

### 4. Continuous Blood Pressure Monitoring
- Real-time SBP, DBP, MAP estimation (per-second updates)  
- Drift-corrected long-term BP tracking  
- Reliable even during motion or posture changes  
- Noninvasive & wearable-friendly

---

### 5. Personalized Modeling
- Individual baseline calibration (SBP/DBP + PTT)  
- PTT–BP relationship optimized per user  
- Learns and updates user-specific BP trends  
- Suitable for long-term monitoring applications

---

## System Workflow

### 1. Wearable Device Setup
- Device capable of simultaneous **ECG + PPG** acquisition  
- IMU sensor recommended for motion robustness  
- Sampling rates typically:
  - ECG: ≥128 Hz  
  - PPG: 100–128 Hz  
  - IMU: 25–100 Hz  

---

### 2. Initialization (Baseline Calibration)
1. User measures BP using a cuff-based device (SBP/DBP).  
2. ECG–PPG signals are recorded simultaneously.  
3. Initial PTT and BP values generate the baseline calibration function.

---

### 3. Real-Time Data Collection & Monitoring
Pipeline:
