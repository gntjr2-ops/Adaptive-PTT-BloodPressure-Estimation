# sqi.py
import numpy as np

def sqi_ptt(ptt_series, imu=None):
    """
    간단 SQI: 분산/이상치 및 (선택) IMU 활동량 반영
    0~1 범위, 높을수록 양호
    """
    if ptt_series is None or len(ptt_series) == 0:
        return 0.0
    ptt = np.asarray(ptt_series, float)
    ptt = ptt[np.isfinite(ptt)]
    if len(ptt) == 0: return 0.0

    # 변동성 기반 (IQR 작을수록 좋음)
    q1, q3 = np.percentile(ptt, 25), np.percentile(ptt, 75)
    iqr = q3 - q1
    # 50~250ms 범위 기대 (임의 안전 범위)
    in_range = np.mean((ptt > 0.05) & (ptt < 0.4))
    # 스코어 결합
    v = np.exp(-iqr / 0.03) * in_range

    if imu is not None and len(imu) > 0:
        # IMU 활동량 높을수록 SQI 페널티
        a = np.asarray(imu, float)
        if a.ndim == 2:  # Nx3 가정
            a = np.sqrt((a**2).sum(axis=1))
        act = np.median(np.abs(a))
        v *= 1.0 / (1.0 + 5.0*act)  # 임의 페널티
    return float(np.clip(v, 0.0, 1.0))
