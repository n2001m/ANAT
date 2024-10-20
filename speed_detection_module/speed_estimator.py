import numpy as np


class SpeedEstimator:
    def __init__(self, ipm_height, ipm_width, fps, meters_per_pixel, line1_y, line2_y):
        self.ipm_height = ipm_height
        self.ipm_width = ipm_width
        self.fps = fps
        self.meters_per_pixel = meters_per_pixel

        self.line1_y = line1_y
        self.line2_y = line2_y

        self.real_world_distance = self.calculate_real_world_distance()
        self.vehicle_positions = {}

    def calculate_real_world_distance(self):
        return abs(self.line2_y - self.line1_y) * self.meters_per_pixel

    def estimate_speed(self, track_id, ipm_position, depth, frame_number):
        current_time = frame_number / self.fps

        if track_id not in self.vehicle_positions:
            self.vehicle_positions[track_id] = {'last_position': ipm_position, 'last_time': current_time,
                                                'last_depth': depth}
            return None

        last_position = self.vehicle_positions[track_id]['last_position']
        last_time = self.vehicle_positions[track_id]['last_time']
        last_depth = self.vehicle_positions[track_id]['last_depth']

        # Calculate distance traveled in IPM space
        distance_pixels = np.linalg.norm(ipm_position - last_position)
        distance_meters = distance_pixels * self.meters_per_pixel

        # Incorporate depth change
        depth_change = abs(depth - last_depth)

        # Pythagoras theorem to combine horizontal and vertical movement
        total_distance = np.sqrt(distance_meters ** 2 + depth_change ** 2)

        # Calculate time difference
        time_diff = current_time - last_time

        if time_diff > 0:
            speed_mps = total_distance / time_diff  # Speed in meters per second
            speed_km_h = speed_mps * 3.6  # Convert to km/h

            # Update position, depth, and time
            self.vehicle_positions[track_id] = {'last_position': ipm_position, 'last_time': current_time,
                                                'last_depth': depth}

            return speed_km_h

        return None

    def get_line_positions(self):
        return self.line1_y, self.line2_y