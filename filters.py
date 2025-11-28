# filters.py
import numpy as np
from scipy.signal import butter, filtfilt

def _safe_filtfilt(b, a, x):
    if x is None:
        return None
    x = np.asarray(x, float)
    padlen = 3 * max(len(a), len(b))
    if x.ndim == 1 and x.size <= padlen:
        return x.copy()
    if x.ndim == 2:
        return np.vstack([_safe_filtfilt(b, a, x[:, i]) for i in range(x.shape[1])]).T
    return filtfilt(b, a, x)

def bandpass(x, fs, lo, hi, order=3):
    lo = max(lo, 1e-3)
    hi = min(hi, fs/2 - 1e-3)
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype="band")
    return _safe_filtfilt(b, a, x)

def highpass(x, fs, fc, order=3):
    fc = max(fc, 1e-3)
    b, a = butter(order, fc/(fs/2), btype="high")
    return _safe_filtfilt(b, a, x)

def lowpass(x, fs, fc, order=3):
    fc = min(fc, fs/2 - 1e-3)
    b, a = butter(order, fc/(fs/2), btype="low")
    return _safe_filtfilt(b, a, x)

def movavg(x, k):
    x = np.asarray(x, float)
    k = max(1, int(k))
    if k <= 1:
        return x
    w = np.ones(k)/k
    return np.convolve(x, w, mode="same")

def zscore(x, eps=1e-9):
    x = np.asarray(x, float)
    m = np.mean(x)
    s = np.std(x)
    return (x - m) / (s + eps)

def diff1(x):
    x = np.asarray(x, float)
    return np.diff(x, prepend=x[:1])
