# pipeline.py
import numpy as np
from preprocess import preprocess_ecg, preprocess_ppg, normalize_01
from detect import detect_rpeaks_ecg, detect_ppg_feet, pair_rfoot_to_ptt
from sqi import sqi_ptt
from calibration import AdaptiveCalibrator
from model import Kalman1D

class PTTBPInference:
    """
    ECG+PPG로 PTT 산출 → 적응형 교정 → SBP/DBP 연속 추정
    """
    def __init__(self, fs_ecg=128, fs_ppg=128, win_sec=10.0,
                 init_sbp=None, init_dbp=None, init_ptt=None):
        self.fs_ecg = fs_ecg
        self.fs_ppg = fs_ppg
        self.win_sec = win_sec

        # SBP, DBP용 교정기
        self.cal_sbp = AdaptiveCalibrator(init_bp=init_sbp, init_ptt=init_ptt, lr=0.01, lam=1e-3)
        self.cal_dbp = AdaptiveCalibrator(init_bp=init_dbp, init_ptt=init_ptt, lr=0.01, lam=1e-3)

        # 칼만 필터
        self.kf_sbp = Kalman1D(x0=init_sbp, q=2.0, r=9.0)
        self.kf_dbp = Kalman1D(x0=init_dbp, q=2.0, r=9.0)

    def _robust_summary(self, arr):
        arr = np.asarray(arr, float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0: return None
        return float(np.median(arr))

    def process_window(self, ecg, ppg, imu=None, cuff_ref=None):
        """
        cuff_ref: dict 예) {"SBP": 122, "DBP": 78}
        """
        # 1) 전처리
        ecg_f = preprocess_ecg(ecg, self.fs_ecg)
        ppg_f = preprocess_ppg(ppg, self.fs_ppg)
        ppg_n = normalize_01(ppg_f)

        # 2) 이벤트 검출
        rpk = detect_rpeaks_ecg(ecg_f, self.fs_ecg, min_rr=0.3)
        feet = detect_ppg_feet(ppg_n, self.fs_ppg, min_rr=0.3)

        # 3) PTT 계산 및 품질
        ptts = pair_rfoot_to_ptt(rpk, feet, self.fs_ecg, self.fs_ppg, max_delay_s=0.6)
        ptt_med = self._robust_summary(ptts)
        sqi = sqi_ptt(ptts, imu=imu)

        # 4) 적응형 보정 업데이트
        #   - cuff_ref가 있을 때 지도학습 (주기적 커프 캘리브레이션 시 활용)
        self.cal_sbp.add_point(ptt_med, cuff_ref.get("SBP") if cuff_ref else None, weight=1.0)
        self.cal_dbp.add_point(ptt_med, cuff_ref.get("DBP") if cuff_ref else None, weight=1.0)
        #   - 주기적 재적합 (옵션)
        if cuff_ref:
            self.cal_sbp.refit_from_history()
            self.cal_dbp.refit_from_history()

        # 5) BP 예측 및 칼만 스무딩
        sbp_raw = self.cal_sbp.predict(ptt_med)
        dbp_raw = self.cal_dbp.predict(ptt_med)
        sbp = self.kf_sbp.update(sbp_raw)
        dbp = self.kf_dbp.update(dbp_raw)
        map_bp = None
        if sbp is not None and dbp is not None:
            map_bp = 1/3*sbp + 2/3*dbp

        return {
            "PTT_median": ptt_med,
            "PTT_all": ptts,
            "SQI": sqi,
            "SBP_raw": sbp_raw, "DBP_raw": dbp_raw,
            "SBP": sbp, "DBP": dbp, "MAP": map_bp,
            "N_R": int(len(rpk)), "N_FOOT": int(len(feet))
        }
