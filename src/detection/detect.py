# detect.py
import numpy as np
from scipy.signal import find_peaks
from filters import diff1, movavg

def detect_rpeaks_ecg(ecg, fs, min_rr=0.3):
    # 간단 QRS 에너지 기반 peak
    x = ecg**2
    x = movavg(x, int(0.08*fs) or 1)
    height = np.percentile(x, 80)
    distance = int(min_rr * fs)
    peaks, _ = find_peaks(x, height=height, distance=distance)
    return peaks

def detect_ppg_feet(ppg, fs, min_rr=0.3):
    # foot: 1차 미분 최소점(상승 시작) 근처를 양의 peak로 잡기 위해 -d/dt에 peak 탐지
    d1 = diff1(ppg)
    inv = -movavg(d1, int(0.04*fs) or 1)
    height = np.percentile(inv, 75)
    distance = int(min_rr * fs)
    feet, _ = find_peaks(inv, height=height, distance=distance)
    return feet

def pair_rfoot_to_ptt(rpeaks, feet, fs_ecg, fs_ppg, max_delay_s=0.6):
    # R-peak 이후 첫 foot을 순차 매칭
    if rpeaks is None or feet is None or len(rpeaks)==0 or len(feet)==0:
        return []
    i = j = 0
    ptts = []
    lim = max_delay_s
    while i < len(rpeaks) and j < len(feet):
        t_r = rpeaks[i] / fs_ecg
        t_f = feet[j] / fs_ppg
        if t_f <= t_r:
            j += 1; continue
        if t_f - t_r <= lim:
            ptts.append(t_f - t_r)
            i += 1; j += 1
        else:
            i += 1
    return ptts
