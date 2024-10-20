import numpy as np


class SpeedCalculator:
    def __init__(self, smoothing_window=5, max_history=100):
        self.speed_history = {}
        self.smoothing_window = smoothing_window
        self.max_history = max_history

    def update_history(self, track_id, speed):
        if track_id not in self.speed_history:
            self.speed_history[track_id] = []

        self.speed_history[track_id].append(speed)

        if len(self.speed_history[track_id]) > self.max_history:
            self.speed_history[track_id] = self.speed_history[track_id][-self.max_history:]

    def get_smoothed_speed(self, track_id):
        if track_id not in self.speed_history or len(self.speed_history[track_id]) < self.smoothing_window:
            return None

        return np.mean(self.speed_history[track_id][-self.smoothing_window:])

    def calculate_speed_confidence(self, track_id):
        if track_id not in self.speed_history or len(self.speed_history[track_id]) < self.smoothing_window:
            return 0.0

        recent_speeds = self.speed_history[track_id][-self.smoothing_window:]
        speed_std = np.std(recent_speeds)
        speed_mean = np.mean(recent_speeds)

        if speed_mean == 0:
            return 0.0

        coefficient_of_variation = speed_std / speed_mean
        confidence = 1 / (1 + coefficient_of_variation)

        return confidence