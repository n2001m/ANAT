import numpy as np
import cv2


class BoundingBoxConstructor:
    def __init__(self, vanishing_points, camera_matrix):
        self.vanishing_points = vanishing_points
        self.camera_matrix = camera_matrix

    def construct_3d_box(self, bbox_2d, depth, aspect_ratio=None):
        try:
            x1, y1, x2, y2 = bbox_2d
            if x1 >= x2 or y1 >= y2 or depth <= 0:
                raise ValueError("Invalid bounding box or depth")

            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            width = x2 - x1
            height = y2 - y1

            # Estimate 3D dimensions
            focal_length = self.camera_matrix[0, 0]
            width_3d = width * depth / focal_length
            height_3d = height * depth / focal_length

            if aspect_ratio is None:
                length_3d = max(width_3d, height_3d)
            else:
                length_3d = max(width_3d, height_3d) * aspect_ratio

            # Construct 3D bounding box corners
            corners_3d = np.array([
                [-width_3d / 2, -height_3d / 2, length_3d / 2],
                [width_3d / 2, -height_3d / 2, length_3d / 2],
                [width_3d / 2, height_3d / 2, length_3d / 2],
                [-width_3d / 2, height_3d / 2, length_3d / 2],
                [-width_3d / 2, -height_3d / 2, -length_3d / 2],
                [width_3d / 2, -height_3d / 2, -length_3d / 2],
                [width_3d / 2, height_3d / 2, -length_3d / 2],
                [-width_3d / 2, height_3d / 2, -length_3d / 2]
            ])

            # Align with vanishing points
            rotation_matrix = np.column_stack(self.vanishing_points)
            corners_3d = np.dot(corners_3d, rotation_matrix.T)

            # Translate to center position
            corners_3d += np.array([center[0], center[1], depth])

            return corners_3d
        except Exception as e:
            print(f"Error in constructing 3D box: {str(e)}")
            return None

    def project_3d_to_2d(self, points_3d):
        print(f"Projecting 3D points: {points_3d}")
        points_2d, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), self.camera_matrix, None)
        print(f"Projected 2D points: {points_2d}")
        return points_2d.reshape(-1, 2)

    def transform_vp2_vp3(self, point):
        # Transform point from VP2-VP3 plane to image coordinates
        vp2, vp3 = self.vanishing_points[1:3]
        basis = np.column_stack((vp2, vp3, np.cross(vp2, vp3)))
        return np.dot(basis, point)

    def transform_vp1_vp2(self, point):
        # Transform point from VP1-VP2 plane to image coordinates
        vp1, vp2 = self.vanishing_points[:2]
        basis = np.column_stack((vp1, vp2, np.cross(vp1, vp2)))
        return np.dot(basis, point)