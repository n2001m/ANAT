import cv2
import numpy as np
import json

class Masker:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.road_mask = None
        self.points = []

    def manual_road_selection(self, frame):
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Road Selection', frame)

        clone = frame.copy()
        cv2.namedWindow('Road Selection')
        cv2.setMouseCallback('Road Selection', mouse_callback)

        while True:
            cv2.imshow('Road Selection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                frame = clone.copy()
                self.points = []
            elif key == ord('c'):
                break

        cv2.destroyWindow('Road Selection')

        if len(self.points) > 2:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            points = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
            self.road_mask = mask
        else:
            print("Not enough points selected. Using full frame.")
            self.road_mask = np.ones((self.height, self.width), dtype=np.uint8) * 255

    def apply_mask(self, frame):
        if self.road_mask is None:
            raise ValueError("Road mask has not been created. Call manual_road_selection first.")
        return cv2.bitwise_and(frame, frame, mask=self.road_mask)

    def get_mask(self):
        return self.road_mask

    def apply_depth_mask(self, frame, depth_map, threshold=0.5):
        if self.road_mask is None:
            raise ValueError("Road mask has not been created. Call manual_road_selection first.")

        if frame.shape[:2] != (self.height, self.width) or depth_map.shape != (self.height, self.width):
            raise ValueError("Frame or depth map dimensions do not match the initialized dimensions.")

        # Normalize depth map to 0-1 range
        normalized_depth = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX)

        # Create binary mask based on depth threshold
        depth_mask = (normalized_depth > threshold).astype(np.uint8) * 255

        # Combine depth mask with road mask
        combined_mask = cv2.bitwise_and(self.road_mask, depth_mask)

        # Apply combined mask to frame
        return cv2.bitwise_and(frame, frame, mask=combined_mask)

    def save_road_mask(self, filename):
        np.save(filename, self.road_mask)

    def load_road_mask(self, filename):
        self.road_mask = np.load(filename)