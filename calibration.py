# calibration.py
import numpy as np
from collections import deque

class AdaptiveCalibrator:
    """
    BP = a * (1/PTT) + b  형태의 개인화 보정 모델
    - 초기 cuff 참값으로 a,b 초기화
    - 슬라이딩/지수 가중 업데이트 (forgetting)
    - SBP / DBP 각각 별도 인스턴스로 운용 권장
    """
    def __init__(self, init_bp=None, init_ptt=None, max_hist=300, lr=0.01, lam=1e-3):
        self.a = None
        self.b = None
        self.hist = deque(maxlen=max_hist)
        self.lr = lr      # 온라인 업데이트 학습률
        self.lam = lam    # L2 정규화
        if init_bp is not None and init_ptt is not None and init_ptt > 1e-6:
            inv = 1.0 / init_ptt
            # 초기값: b=BP- a*inv. a는 경험치로 작게 시작
            self.a = 10.0     # 대략적 시작 (개인별 조정 필요)
            self.b = float(init_bp - self.a*inv)

    def add_point(self, ptt, bp_ref=None, weight=1.0):
        if ptt is None or ptt <= 0:
            return
        inv = 1.0 / float(ptt)
        self.hist.append((inv, bp_ref, weight))
        # bp_ref가 있을 때는 지도 신호로 업데이트
        if bp_ref is not None:
            self._sgd_step(inv, float(bp_ref), weight)

    def _sgd_step(self, inv, bp, w):
        if self.a is None or self.b is None:
            self.a, self.b = 10.0, bp - 10.0*inv
        y = self.a*inv + self.b
        err = (y - bp)
        # L2 정규화 포함 경사 하강
        self.a -= self.lr * (w * err * inv + self.lam * self.a)
        self.b -= self.lr * (w * err + self.lam * self.b)

    def refit_from_history(self):
        # 주기적으로 과거 히스토리로 재적합 (bp_ref가 있는 점만 사용)
        xs, ys, ws = [], [], []
        for inv, bp, w in self.hist:
            if bp is None: 
                continue
            xs.append([inv, 1.0]); ys.append(bp); ws.append(w)
        if len(xs) < 2:
            return
        X = np.asarray(xs, float)
        y = np.asarray(ys, float)
        W = np.diag(ws) if len(ws)==len(xs) else np.eye(len(xs))
        # (X^T W X + lam I)^{-1} X^T W y
        lamI = self.lam * np.eye(X.shape[1])
        beta = np.linalg.pinv(X.T @ W @ X + lamI) @ (X.T @ W @ y)
        self.a, self.b = float(beta[0]), float(beta[1])

    def predict(self, ptt):
        if ptt is None or ptt <= 0 or self.a is None or self.b is None:
            return None
        inv = 1.0 / float(ptt)
        return float(self.a * inv + self.b)
