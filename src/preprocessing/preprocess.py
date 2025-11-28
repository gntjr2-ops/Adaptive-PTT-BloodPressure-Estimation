# preprocess.py
import numpy as np
from filters import bandpass, highpass, lowpass, movavg, zscore

def preprocess_ecg(ecg, fs):
    # QRS 대역(5~20/30Hz) 강조
    x = bandpass(ecg, fs, 5.0, 20.0, order=3)
    return x

def preprocess_ppg(ppg, fs):
    # 드리프트 제거 + 대역통과(0.5~8Hz) + 약한 평활
    x = highpass(ppg, fs, 0.05, order=2)
    x = bandpass(x, fs, 0.5, 8.0, order=3)
    x = movavg(x, int(0.03*fs) or 1)
    return x

def normalize_01(x):
    x = np.asarray(x, float)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

def clip_iqr(x, q=3.0):
    x = np.asarray(x, float)
    q1, q3 = np.percentile(x, 25), np.percentile(x, 75)
    iqr = q3 - q1
    lo, hi = q1 - q*iqr, q3 + q*iqr
    return np.clip(x, lo, hi)
