# ruff: noqa
import cv2
import numpy as np


class PerspectiveCorrector:
    """
    Handles perspective correction ensuring the reference object (e.g., coin)
    becomes a perfect circle in a top-down view.
    """

    def __init__(self):
        pass

    def find_homography(self, image, reference_mask):
        """
        Calculates the Homography matrix to warp the image such that the
        elliptical reference object in the mask becomes circular.

        Args:
            image: Original input image (H, W, 3)
            reference_mask: Binary mask of the reference object (H, W)

        Returns:
            H (numpy.ndarray): 3x3 Homography matrix
            warped_size (tuple): Suggested (width, height) for the warped image
            coin_info (tuple): (center, radius) of the restored coin in the warped image
        """
        # 1. Find Contour of the reference object
        # Ensure mask is uint8
        if reference_mask.dtype != np.uint8:
            reference_mask = reference_mask.astype(np.uint8)

        contours, _ = cv2.findContours(reference_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("PerspectiveCorrector: No contour found in reference mask.")
            return None, None, None, None

        largest_contour = max(contours, key=cv2.contourArea)

        # 2. Fit Ellipse
        if len(largest_contour) < 5:
            print("PerspectiveCorrector: Contour too small for ellipse fitting.")
            return None, None, None, None

        try:
            # (center(x, y), (MA, ma), angle)
            # axes lengths are the full lengths of the axes (diameter-like)
            fitted_ellipse = cv2.fitEllipse(largest_contour)
            (cx, cy), (d1, d2), angle = fitted_ellipse
        except Exception as e:
            print(f"PerspectiveCorrector: Ellipse fitting failed - {e}")
            return None, None, None, None

        # Determine Major and Minor axes
        # We assume the larger one is the Major axis (diameter in 3D, approx)
        if d1 > d2:
            major_axis = d1
            minor_axis = d2
            # Angle returned by fitEllipse is for d1?
            # OpenAI docs: angle is the rotation of the first axis (d1) from horizontal?
            # Actually, OpenCV documentation says angle is in degrees.
            # (cx, cy) is center. (width, height) are axes.
            # The angle is the rotation of the rectangle in clockwise direction.
        else:
            major_axis = d2
            minor_axis = d1
            # Using the geometry logic below, we reconstruct points.

        # Strategy: Map 4 key points on the ellipse to 4 points on a circle.
        # But `fitEllipse` gives us the parametric form directly.
        # We can construct source points from the ellipse parameters.

        theta_rad = np.radians(angle)

        # Ellipse Extrema points along axes (Source)
        # Using parametric equation relative to center:
        # P = center + R(angle) * [dx, dy]
        # d1 is coupled with local x, d2 with local y?
        # OpenCV fitEllipse returns (width, height) of the rotated bounding rect roughly.
        # Let's use the explicit standard form.

        # Easier approach: Use the properties of the axes directly.
        # Major axis endpoints and Minor axis endpoints.

        # Half aces
        a = d1 / 2.0
        b = d2 / 2.0

        # Rotation matrix for the ellipse angle
        R = np.array(
            [[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]]
        )

        # Local canonical points (center at 0,0)
        # d1 is the length of the axis tilted by 'angle'. Let's call this local X.
        p1_local = np.array([a, 0])
        p2_local = np.array([-a, 0])
        # d2 is the length of the perpendicular axis. Local Y.
        p3_local = np.array([0, b])
        p4_local = np.array([0, -b])

        # Rotated and translated to global
        p1 = np.dot(R, p1_local) + np.array([cx, cy])
        p2 = np.dot(R, p2_local) + np.array([cx, cy])
        p3 = np.dot(R, p3_local) + np.array([cx, cy])
        p4 = np.dot(R, p4_local) + np.array([cx, cy])

        src_pts = np.array([p1, p2, p3, p4], dtype=np.float32)

        # Target: A perfect circle with radius = Major Axis / 2
        # Use the larger dimension to preserve resolution
        target_radius = max(a, b)

        # Target points (Canonical Circle)
        # We map p1, p2 (derived from d1) to the local X axis of the target circle
        # We map p3, p4 (derived from d2) to the local Y axis of the target circle

        tp1_local = np.array([target_radius, 0])
        tp2_local = np.array([-target_radius, 0])
        tp3_local = np.array([0, target_radius])
        tp4_local = np.array([0, -target_radius])

        # Apply the SAME Rotation R to target points to keep orientation
        tp1 = np.dot(R, tp1_local) + np.array([cx, cy])
        tp2 = np.dot(R, tp2_local) + np.array([cx, cy])
        tp3 = np.dot(R, tp3_local) + np.array([cx, cy])
        tp4 = np.dot(R, tp4_local) + np.array([cx, cy])

        dst_pts = np.array([tp1, tp2, tp3, tp4], dtype=np.float32)

        # Compute Homography
        H, _ = cv2.findHomography(src_pts, dst_pts)

        return H, (image.shape[1], image.shape[0]), (cx, cy, target_radius), fitted_ellipse

    def warp_image(self, image, H, target_size):
        """
        Applies the homography to the image.
        """
        return cv2.warpPerspective(image, H, target_size)

    def warp_point(self, point, H):
        """
        Warps a single point (x, y).
        """
        src = np.array([[[point[0], point[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)
        return dst[0][0]
