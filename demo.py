# demo.py
import numpy as np
from pipeline import PTTBPInference

# ---------------- 합성기(간단) ----------------
def synth_rr_series(hr_bpm, win_sec, sdnn=0.04, seed=0):
    rng = np.random.default_rng(seed)
    n_beats = int(round(hr_bpm/60.0 * win_sec))
    base_rr = 60.0 / hr_bpm
    t = np.linspace(0, win_sec, n_beats, endpoint=False)
    rr = base_rr + rng.normal(scale=sdnn, size=n_beats)
    rr = np.clip(rr, 0.4, 1.5)
    return rr

def rr_to_peaks(rr, fs):
    tt = np.cumsum(rr); tt -= tt[0]
    idx = np.unique(np.maximum(0, np.round(tt*fs).astype(int)))
    return idx

def synth_signals(fs_ecg=128, fs_ppg=128, win_sec=20, 
                  hr_bpm=72, base_ptt=0.25, bp_profile=None, seed=0):
    """
    ECG: R-peak 가우시안
    PPG: foot = r_peak + ptt(t), 이후 지수감쇠 맥파
    bp_profile: 길이 win_sec (초 단위) SBP 트렌드 (예: 스트레스 상승 등)
    """
    rng = np.random.default_rng(seed)
    rr = synth_rr_series(hr_bpm, win_sec, sdnn=0.03, seed=seed)
    rpk = rr_to_peaks(rr, fs_ecg)

    N_e = fs_ecg*win_sec
    ecg = np.zeros(N_e, float)
    w = int(0.02*fs_ecg)
    k = np.arange(-w, w+1)
    gauss = np.exp(-0.5*(k/(0.007*fs_ecg))**2)
    for rp in rpk:
        if rp-w < 0 or rp+w >= N_e: continue
        ecg[rp-w:rp+w+1] += gauss
    ecg += 0.01*rng.standard_normal(N_e)

    # BP -> PTT 관계를 단순화 (PTT ≈ c0 - c1*SBP 변화)
    if bp_profile is None:
        # 20초 동안 SBP가 120→130 선형 증가 시나리오
        t = np.linspace(0, win_sec, win_sec, endpoint=False)
        bp_profile = 120 + 10*(t/ max(1, win_sec-1))
    else:
        bp_profile = np.asarray(bp_profile, float)
        if len(bp_profile) != win_sec:
            bp_profile = np.interp(np.arange(win_sec), np.linspace(0, win_sec-1, len(bp_profile)), bp_profile)

    sbp_per_sec = np.repeat(bp_profile, fs_ppg)  # 초당

    # PTT(t) = base_ptt - alpha*(SBP-120)
    alpha = 0.0015
    ptt_t = base_ptt - alpha * (sbp_per_sec - 120.0)
    ptt_t = np.clip(ptt_t, 0.12, 0.35)

    N_p = fs_ppg*win_sec
    ppg = np.zeros(N_p, float)
    tail = int(0.35*fs_ppg)
    # rpk from ECG -> map to PPG sampling
    rpk_ppg = (rpk / fs_ecg * fs_ppg).astype(int)
    for rp in rpk_ppg:
        if rp >= N_p: continue
        ptt_here = ptt_t[min(rp, len(ptt_t)-1)]
        foot = rp + int(round(ptt_here * fs_ppg))
        if foot >= N_p: continue
        end = min(N_p, foot + tail)
        idx = np.arange(end - foot)
        wave = np.exp(-idx/(0.08*fs_ppg))
        ppg[foot:end] += wave
    # 정규화 + 잡음
    ppg = (ppg - ppg.min()) / (ppg.max()-ppg.min()+1e-9)
    ppg += 0.01*rng.standard_normal(N_p)

    # 간단 IMU (움직임: 중간 정도)
    imu = 0.02*rng.standard_normal((win_sec*32, 3))
    return ecg.astype(np.float32), ppg.astype(np.float32), imu.astype(np.float32), rpk

# ---------------- 데모 ----------------
if __name__ == "__main__":
    fsE, fsP = 128, 128
    win = 20

    # 초기 캘리브레이션: 컵 혈압 (SBP/DBP)와 동시 측정된 PTT (예: 0.25s)
    init_sbp, init_dbp, init_ptt = 120.0, 78.0, 0.25
    pipe = PTTBPInference(fs_ecg=fsE, fs_ppg=fsP, win_sec=win,
                          init_sbp=init_sbp, init_dbp=init_dbp, init_ptt=init_ptt)

    # 1) 안정 구간 (낮은 SBP)
    ecg, ppg, imu, rpk = synth_signals(fsE, fsP, win_sec=win, hr_bpm=72, base_ptt=0.27, seed=1)
    res = pipe.process_window(ecg, ppg, imu=imu, cuff_ref={"SBP": 120, "DBP": 78})
    print("=== Segment 1 (baseline) ===")
    print(f"PTT_med={res['PTT_median']:.3f}s | SQI={res['SQI']:.2f} | "
          f"SBP={res['SBP']:.1f} ({res['SBP_raw']:.1f}) | DBP={res['DBP']:.1f} ({res['DBP_raw']:.1f}) | MAP={res['MAP']:.1f}")

    # 2) 가벼운 상승(운동/스트레스) → SBP 상승 → PTT 감소
    ecg, ppg, imu, rpk = synth_signals(fsE, fsP, win_sec=win, hr_bpm=78, base_ptt=0.24, seed=2)
    res = pipe.process_window(ecg, ppg, imu=imu, cuff_ref=None)  # cuff 없는 보정
    print("=== Segment 2 (mild up) ===")
    print(f"PTT_med={res['PTT_median']:.3f}s | SQI={res['SQI']:.2f} | "
          f"SBP={res['SBP']:.1f} ({res['SBP_raw']:.1f}) | DBP={res['DBP']:.1f} ({res['DBP_raw']:.1f}) | MAP={res['MAP']:.1f}")

    # 3) 더 상승 + 커프 재교정 포인트 도입
    ecg, ppg, imu, rpk = synth_signals(fsE, fsP, win_sec=win, hr_bpm=82, base_ptt=0.21, seed=3)
    res = pipe.process_window(ecg, ppg, imu=imu, cuff_ref={"SBP": 135, "DBP": 82})
    print("=== Segment 3 (higher + cuff) ===")
    print(f"PTT_med={res['PTT_median']:.3f}s | SQI={res['SQI']:.2f} | "
          f"SBP={res['SBP']:.1f} ({res['SBP_raw']:.1f}) | DBP={res['DBP']:.1f} ({res['DBP_raw']:.1f}) | MAP={res['MAP']:.1f}")
