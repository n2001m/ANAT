import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import random

class CameraCalibration:
    def __init__(self, model_path='models/vp_using_seg_model_best.keras'):
        self.focal_length = None
        self.principal_point = None
        self.vanishing_points = None
        self.model_path = model_path
        self.model = None
        self.ipm_matrix = None
        self.width = None
        self.height = None

    def load_model(self):
        try:
            if self.model is None:
                self.model = keras.models.load_model(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def calibrate_camera(self, frames):
        self.load_model()
        self.frames = frames
        self.height, self.width = frames[0].shape[:2]
        self.principal_point = (self.width / 2, self.height / 2)

        vps = []
        for frame in frames:
            vp = self.find_vanishing_point(frame)
            if vp is not None:
                vps.append(vp)

        if not vps:
            raise ValueError("No valid vanishing points detected.")

        best_vp = self.ransac_line_fit(np.array(vps), threshold=10, iterations=100)

        vp2 = self.find_vanishing_point(random.choice(frames))
        vp2 = self.orthogonalize_vanishing_points(best_vp, vp2)

        self.focal_length = self.estimate_focal_length(best_vp, vp2)

        vp3 = np.cross(best_vp, vp2)
        vp3 /= np.linalg.norm(vp3)

        self.vanishing_points = [best_vp, vp2, vp3]

        self.compute_ipm_matrix()

        self.visualize_vanishing_points(frames[0], [best_vp, vp2, vp3])

        return self.ipm_matrix

    def ransac_line_fit(self, points, threshold, iterations):
        best_line = None
        best_inliers = 0
        for _ in range(iterations):
            sample = random.sample(range(len(points)), 2)
            p1, p2 = points[sample]

            line = np.cross(p1, p2)
            inliers = 0

            for p in points:
                dist = abs(np.dot(line, p)) / np.linalg.norm(line)
                if dist < threshold:
                    inliers += 1
            if inliers > best_inliers:
                best_inliers = inliers
                best_line = line

        if best_line is not None:
            return best_line / np.linalg.norm(best_line)
        return None

    def orthogonalize_vanishing_points(self, vp1, vp2):
        vp2_ortho = vp2 - np.dot(vp2, vp1) * vp1
        vp2_ortho /= np.linalg.norm(vp2_ortho)
        return vp2_ortho

    def find_vanishing_point(self, frame):
        self.load_model()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=0)

        segmentation, vp = self.model.predict(frame)
        vp = vp[0]

        vp = np.array([vp[0] * self.width, vp[1] * self.height, 1])
        return vp

    def estimate_focal_length(self, vp1, vp2):
        return np.sqrt(abs(np.dot(vp1, vp2)))

    def compute_ipm_matrix(self):
        src_points = np.float32([
            [0, self.height],
            [self.width, self.height],
            [self.width, 0],
            [0, 0]
        ])

        dst_points = np.float32([
            [0, self.height],
            [self.width, self.height],
            [self.width * 0.75, 0],
            [self.width * 0.25, 0]
        ])

        self.ipm_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    def visualize_vanishing_points(self, frame, vanishing_points, output_path=None):
        frame_with_vps = frame.copy()

        def normalize_vp(vp):
            if vp[2] != 0:
                return [vp[0] / vp[2], vp[1] / vp[2]]
            return [vp[0], vp[1]]

        normalized_vps = [normalize_vp(vp) for vp in vanishing_points]

        img_h, img_w = frame.shape[:2]

        for idx, vp in enumerate(normalized_vps):
            x, y = int(vp[0]), int(vp[1])

            if 0 <= x < img_w and 0 <= y < img_h:
                cv2.circle(frame_with_vps, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(frame_with_vps, f'VP{idx+1}', (x + 15, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                x = max(0, min(x, img_w - 1))
                y = max(0, min(y, img_h - 1))
                cv2.arrowedLine(frame_with_vps, (img_w // 2, img_h), (x, y), (0, 255, 0), 2)

        for vp in normalized_vps:
            x, y = int(vp[0]), int(vp[1])
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))

            cv2.line(frame_with_vps, (img_w // 4, img_h), (x, y), (0, 255, 0), 2)
            cv2.line(frame_with_vps, (img_w // 2, img_h), (x, y), (0, 255, 0), 2)
            cv2.line(frame_with_vps, (3 * img_w // 4, img_h), (x, y), (0, 255, 0), 2)

        if output_path:
            cv2.imwrite(output_path, frame_with_vps)
            print(f"Vanishing points visualization saved to {output_path}")
        else:
            cv2.imshow('Vanishing Points Visualization', frame_with_vps)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def get_camera_matrix(self):
        if self.focal_length is None or self.principal_point is None:
            raise ValueError("Camera not calibrated. Call calibrate_camera first.")
        return np.array([
            [self.focal_length, 0, self.principal_point[0]],
            [0, self.focal_length, self.principal_point[1]],
            [0, 0, 1]
        ])

    def save_calibration(self, filename):
        data = {
            'focal_length': self.focal_length.tolist() if isinstance(self.focal_length, np.ndarray) else self.focal_length,
            'principal_point': self.principal_point.tolist() if isinstance(self.principal_point, np.ndarray) else self.principal_point,
            'vanishing_points': [vp.tolist() for vp in self.vanishing_points],
            'ipm_matrix': self.ipm_matrix.tolist(),
            'width': self.width,
            'height': self.height
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_calibration(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        self.focal_length = np.array(data['focal_length'])
        self.principal_point = np.array(data['principal_point'])
        self.vanishing_points = [np.array(vp) for vp in data['vanishing_points']]
        self.ipm_matrix = np.array(data['ipm_matrix'])
        self.width = data['width']
        self.height = data['height']

    def apply_ipm(self, frame):
        if self.ipm_matrix is None:
            raise ValueError("IPM matrix is not initialized. Call calibrate_camera first.")
        return cv2.warpPerspective(frame, self.ipm_matrix, (self.width, self.height))