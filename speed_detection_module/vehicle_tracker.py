import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class VehicleTracker:
    def __init__(self, max_frames_to_skip=10, min_hits=3, max_track_length=30, distance_threshold=50):
        self.tracks = {}
        self.frame_count = 0
        self.max_frames_to_skip = max_frames_to_skip
        self.min_hits = min_hits
        self.max_track_length = max_track_length
        self.track_id_count = 0
        self.distance_threshold = distance_threshold

    def update(self, detections):
        self.frame_count += 1
        print(f"Frame {self.frame_count}: Received {len(detections)} detections")

        # Predict new states for all tracks
        for track in self.tracks.values():
            track['kf'].predict()

        # Match detections to tracks
        if len(detections) > 0 and len(self.tracks) > 0:
            # Compute distance matrix
            distance_matrix = np.zeros((len(detections), len(self.tracks)))
            for i, detection in enumerate(detections):
                detection_center = np.mean(detection, axis=0)
                for j, (track_id, track) in enumerate(self.tracks.items()):
                    predicted_pos = track['kf'].x[:3].flatten()
                    distance_matrix[i, j] = np.linalg.norm(detection_center - predicted_pos)

            # Perform assignment
            detection_indices, track_indices = linear_sum_assignment(distance_matrix)

            # Update matched tracks and create new tracks for unmatched detections
            matched_indices = list(zip(detection_indices, track_indices))
            unmatched_detections = set(range(len(detections))) - set(detection_indices)
            unmatched_tracks = set(range(len(self.tracks))) - set(track_indices)

            for d, t in matched_indices:
                if distance_matrix[d, t] < self.distance_threshold:
                    self.update_track(list(self.tracks.keys())[t], detections[d])
                else:
                    self.create_new_track(detections[d])

            for d in unmatched_detections:
                self.create_new_track(detections[d])

            for t in unmatched_tracks:
                track_id = list(self.tracks.keys())[t]
                self.tracks[track_id]['missed_frames'] += 1

        elif len(detections) > 0:
            # If there are no existing tracks, create new tracks for all detections
            for detection in detections:
                self.create_new_track(detection)

        # Remove old tracks
        for track_id in list(self.tracks.keys()):
            if self.frame_count - self.tracks[track_id]['last_seen'] > self.max_frames_to_skip or \
                    self.frame_count - self.tracks[track_id]['first_seen'] > self.max_track_length:
                print(f"Removing track {track_id} due to age or long absence")
                del self.tracks[track_id]

        print(f"Frame {self.frame_count}: Tracking {len(self.tracks)} vehicles")
        for track_id, track in self.tracks.items():
            print(f"Track {track_id}: position = {track['kf'].x[:3].T}, velocity = {track['kf'].x[3:6].T}")

        return self.tracks

    def create_new_track(self, detection):
        self.track_id_count += 1
        kf = KalmanFilter(dim_x=6, dim_z=3)  # 3D position and velocity
        kf.F = np.array([[1, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])  # state transition matrix
        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])  # measurement function
        kf.R *= 10  # measurement uncertainty
        kf.Q *= 0.1  # process uncertainty

        # Use the center of the 3D bounding box as the initial position
        initial_pos = np.mean(detection, axis=0)
        kf.x[:3] = initial_pos.reshape(3, 1)  # initial state (position)
        kf.x[3:] = 0  # initial velocity

        self.tracks[self.track_id_count] = {
            'kf': kf,
            'bbox_3d': detection,
            'last_seen': self.frame_count,
            'first_seen': self.frame_count,
            'missed_frames': 0,
            'hits': 1
        }
        print(f"Created new track {self.track_id_count} at position {kf.x[:3].T}")

    def update_track(self, track_id, detection):
        # Update Kalman filter with the center of the 3D bounding box
        detection_center = np.mean(detection, axis=0)
        self.tracks[track_id]['kf'].update(detection_center.reshape(3, 1))

        self.tracks[track_id]['bbox_3d'] = detection
        self.tracks[track_id]['last_seen'] = self.frame_count
        self.tracks[track_id]['missed_frames'] = 0
        self.tracks[track_id]['hits'] += 1
        print(f"Updated track {track_id}: new position = {self.tracks[track_id]['kf'].x[:3].T}")