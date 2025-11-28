# model.py
class Kalman1D:
    """
    1차원 칼만 필터로 연속 BP 시계열을 안정화
    x_k = x_{k-1} + w,  y_k = x_k + v
    """
    def __init__(self, x0=None, p0=100.0, q=1.0, r=16.0):
        self.x = x0
        self.P = p0
        self.Q = q  # process noise
        self.R = r  # measurement noise

    def update(self, z):
        # predict
        self.P = self.P + self.Q
        # update
        if z is None:
            return self.x
        K = self.P / (self.P + self.R)
        if self.x is None:
            self.x = z
        else:
            self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x
