import cv2
import numpy as np
import json
import os
import logging
from speed_detection_module.camera_calibration import CameraCalibration
from speed_detection_module.car_detection import CarDetection
from speed_detection_module.bounding_box_constructor import BoundingBoxConstructor
from speed_detection_module.vehicle_tracker import VehicleTracker
from speed_detection_module.speed_calculator import SpeedCalculator
from speed_detection_module.depth_estimation import DepthEstimationModel
from speed_detection_module.masker import Masker
from speed_detection_module.optical_flow_estimator import OpticalFlowEstimator
from ultralytics import YOLO
from anpr_module.Help_util import assign_car, preprocess_frame, read_license_plate, write_csv
from google.cloud import storage
from datetime import datetime
import threading
import queue
import argparse
import yaml
import re


class IntegratedVideoProcessor:
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize variables from config
        self.video_path = self.config['video_path']
        self.calibration_file = self.config['calibration_file']
        self.road_mask_file = self.config['road_mask_file']
        self.detection_confidence = self.config['detection_confidence']
        self.speed_threshold = self.config['speed_threshold']
        self.gcs_bucket_name = self.config['gcs_bucket_name']

        try:
            self.license_plate_detector = YOLO(self.config['LP_model'])
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {str(e)}")
            raise

        # Initialize other components
        self.calibration = CameraCalibration()
        self.car_detection = CarDetection()
        self.depth_model = DepthEstimationModel()
        self.tracker = VehicleTracker(max_frames_to_skip=10, min_hits=3, max_track_length=30)
        self.speed_calculator = SpeedCalculator(smoothing_window=5, speed_confidence_threshold=0.8, max_history=100)
        self.flow_estimator = OpticalFlowEstimator()

        # Initialize Google Cloud Storage
        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.gcs_bucket_name)
        except Exception as e:
            self.logger.error(f"Failed to initialize GCS: {str(e)}")
            raise

        # Video capture setup
        self.cap = cv2.VideoCapture(self.video_path)
        self.ret, self.frame = self.cap.read()
        if not self.ret:
            raise ValueError("Failed to read the video file.")
        self.height, self.width = self.frame.shape[:2]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.masker = Masker(self.height, self.width)
        self.ipm_matrix = None
        self.bbox_constructor = None

        self.results = []
        self.license_plate_results = {}

        # Threading setup
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue()

    def _collect_calibration_frames(self, num_frames=10):
        calibration_frames = []
        for _ in range(num_frames):
            ret, frame = self.cap.read()
            if ret:
                calibration_frames.append(frame)
            else:
                break
        return calibration_frames

    def save_image_to_gcs(self, image, gcs_path):
        """Saves an image directly to GCS."""
        try:
            _, encoded_image = cv2.imencode('.jpg', image)
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_string(encoded_image.tobytes(), content_type='image/jpeg')
            self.logger.info(f'Image saved to {gcs_path}')
        except Exception as e:
            self.logger.error(f"Failed to save image to GCS: {str(e)}")

    def draw_3d_box(self, img, corners, color=(0, 255, 0)):
        def draw_line(start, end):
            cv2.line(img, tuple(map(int, start)), tuple(map(int, end)), color, 2)

        # Draw bottom face
        for i in range(4):
            draw_line(corners[i], corners[(i + 1) % 4])

        # Draw top face
        for i in range(4):
            draw_line(corners[i + 4], corners[(i + 1) % 4 + 4])

        # Draw vertical lines
        for i in range(4):
            draw_line(corners[i], corners[i + 4])

        return img

    def process_frame(self):
        while True:
            frame_data = self.frame_queue.get()
            if frame_data is None:
                break

            frame, frame_count, current_time = frame_data
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            vis_frame = frame.copy()
            vis_ipm_frame = self.calibration.apply_ipm(frame.copy())

            # Vehicle Detection and Speed Calculation
            masked_frame = self.masker.apply_mask(frame)

            ipm_frame = self.calibration.apply_ipm(masked_frame)

            depth_map = self.depth_model.estimate_depth(ipm_frame)

            flow = self.flow_estimator.estimate_flow(masked_frame)

            vehicle_detections = self.car_detection.detect_cars(ipm_frame, self.ipm_matrix, self.detection_confidence)

            # ANPR - License Plate Detection
            license_plate_detections = self.license_plate_detector(frame)[0]
            vehicles_detected = []

            bboxes_3d = []
            for det in vehicle_detections:
                x1, y1, x2, y2, conf, cls = det
                center_depth = np.mean(depth_map[int(y1):int(y2), int(x1):int(x2)])
                if np.isnan(center_depth) or np.isinf(center_depth):
                    print(f"Warning: Invalid depth value for detection {det}")
                    continue
                bbox_3d = self.bbox_constructor.construct_3d_box([x1, y1, x2, y2], center_depth, aspect_ratio=1.5)
                if bbox_3d is not None:
                    bboxes_3d.append(bbox_3d)
                    vehicles_detected.append([x1, y1, x2, y2, None, conf])

            # Tracking and Speed Calculation
            tracks = self.tracker.update(bboxes_3d)
            for track_id, track in tracks.items():
                corners_3d = track['bbox_3d']
                corners_2d = self.bbox_constructor.project_3d_to_2d(corners_3d)

                # Draw 3D bounding box in original frame
                self.draw_3d_box(vis_frame, corners_2d, color=(0, 255, 0))

                current_position = np.mean(corners_3d, axis=0)
                speed, confidence = self.speed_calculator.calculate_speed(
                    track_id, current_position, frame_count, self.fps, unit='km/h'
                )

                # Save car image to GCS
                car_image = frame[int(corners_3d[0][1]):int(corners_3d[1][1]),
                            int(corners_3d[0][0]):int(corners_3d[1][0])]
                car_image_gcs_path = f'car_images/car_{track_id}_{timestamp}.jpg'
                self.save_image_to_gcs(car_image, car_image_gcs_path)

                # Check if speed exceeds threshold
                speed_alert = speed > self.speed_threshold if speed is not None else False

                # Associate License Plates with Vehicles
                for license_plate in license_plate_detections.boxes.data.tolist():
                    x1_lp, y1_lp, x2_lp, y2_lp, score_lp, _ = license_plate
                    x1_v, y1_v, x2_v, y2_v, car_id = assign_car(license_plate, vehicles_detected)
                    if car_id != -1:
                        license_plate_image = frame[int(y1_lp):int(y2_lp), int(x1_lp):int(x2_lp)]
                        license_plate_image_gcs_path = f'license_plate_images/license_plate_{car_id}_{timestamp}.jpg'
                        self.save_image_to_gcs(license_plate_image, license_plate_image_gcs_path)

                        lp_crop = preprocess_frame(frame, x1_lp, y1_lp, x2_lp, y2_lp)
                        license_plate_text = read_license_plate(lp_crop)

                        # Save license plate
                        self.license_plate_results[frame_count] = {
                            'track_id': track_id,
                            'license_plate': {'text': license_plate_text, 'bbox': [x1_lp, y1_lp, x2_lp, y2_lp]},
                            'car_image_gcs_path': car_image_gcs_path,
                            'license_plate_image_gcs_path': license_plate_image_gcs_path,
                            'speed': speed,
                            'timestamp': timestamp,
                            'speed_alert': speed_alert
                        }

                # Visualization
                cv2.rectangle(vis_frame, (int(corners_3d[0][0]), int(corners_3d[0][1])),
                              (int(corners_3d[1][0]), int(corners_3d[1][1])), (0, 255, 0), 2)
                cv2.putText(vis_frame, f"ID: {track_id}, Speed: {speed:.2f} km/h",
                            (int(corners_3d[0][0]), int(corners_3d[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Store results
                self.result_queue.put({
                    'frame': frame_count,
                    'track_id': track_id,
                    'speed': speed if speed is not None else 'N/A',
                    'license_plate': self.license_plate_results.get(frame_count, {}).get('license_plate', 'N/A'),
                    'car_image_gcs_path': car_image_gcs_path,
                    'license_plate_image_gcs_path': license_plate_image_gcs_path,
                    'timestamp': timestamp,
                    'confidence': confidence if confidence is not None else 'N/A',
                    'speed_alert': speed_alert
                })

            # Save visualized frame
            vis_frame_gcs_path = f'visualized_frames/frame_{frame_count}.jpg'
            self.save_image_to_gcs(vis_frame, vis_frame_gcs_path)

        self.result_queue.put(None)  # Signal that processing is complete

    def process_video(self):
        # Road mask selection or loading
        if os.path.exists(self.road_mask_file):
            self.masker.load_road_mask(self.road_mask_file)
            print(f"Loaded existing road mask from {self.road_mask_file}")
        else:
            print("Please select the road area...")
            self.masker.manual_road_selection(self.frame)
            self.masker.save_road_mask(self.road_mask_file)
            print(f"Saved road mask to {self.road_mask_file}")

        # Initialize camera calibration
        if os.path.exists(self.calibration_file):
            self.calibration.load_calibration(self.calibration_file)
            self.logger.info(f"Loaded existing camera calibration from {self.calibration_file}")
        else:
            self.logger.info("Performing camera calibration...")
            self.calibration.calibrate_camera(self._collect_calibration_frames())
            self.calibration.save_calibration(self.calibration_file)
            self.logger.info(f"Saved camera calibration to {self.calibration_file}")

        # Reset video capture to the beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Get necessary matrices
        self.ipm_matrix = self.calibration.ipm_matrix
        camera_matrix = self.calibration.get_camera_matrix()
        vanishing_points = self.calibration.vanishing_points

        # Initialize BoundingBoxConstructor with calibration results
        self.bbox_constructor = BoundingBoxConstructor(vanishing_points, camera_matrix)

        frame_count = 0

        # Start processing thread
        processing_thread = threading.Thread(target=self.process_frame)
        processing_thread.start()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            current_time = frame_count / self.fps

            self.frame_queue.put((frame, frame_count, current_time))

        self.frame_queue.put(None)  # Signal end of video
        processing_thread.join()

        # Collect results
        while True:
            result = self.result_queue.get()
            if result is None:
                break
            self.results.append(result)

        # Save results JSON to GCS
        results_json = json.dumps(self.results, indent=2)
        blob = self.bucket.blob('output/speed_estimation_results.json')
        blob.upload_from_string(results_json, content_type='application/json')
        self.logger.info('Results JSON uploaded to GCS.')

        self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Integrated Video Processor for Speed Detection and ANPR")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    processor = IntegratedVideoProcessor(args.config)
    processor.process_video()


if __name__ == "__main__":
    main()
