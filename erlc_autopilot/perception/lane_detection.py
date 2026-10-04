"""
Classical computer-vision lane detector.

Pipeline (the same family of techniques used in real lane-keep-assist
systems and the Udacity/sentdex self-driving-car tutorials, just tuned for
an ER:LC-style road): grayscale + color threshold -> Canny edges -> region
of interest mask -> probabilistic Hough transform -> slope-based left/right
line clustering -> robust polynomial fit -> exponential temporal smoothing
-> lane-center offset & curvature estimate.

This module receives ONLY a BGR image (whatever came out of screen capture
or the simulator renderer). It never touches any ground-truth world state.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class LaneResult:
    offset_px: float          # + means car is right of lane center
    offset_norm: float        # offset_px normalized by half the detected lane width [-1, 1]
    curvature: float          # unitless signed curvature estimate (+ = curving right)
    confidence: float         # 0..1, how much we trust this frame's detection
    left_line: Optional[tuple]   # (x_bottom, y_bottom, x_top, y_top, max_y_seen)
    right_line: Optional[tuple]
    debug: np.ndarray


class LaneDetector:
    MIN_LANE_PX = 120   # implausibly narrow -> probably a bad/short extrapolated fit
    MAX_LANE_PX = 560   # implausibly wide -> same

    def __init__(self, smoothing: int = 7):
        self._offset_hist = deque(maxlen=smoothing)
        self._curve_hist = deque(maxlen=smoothing)
        self._last_left = None
        self._last_right = None
        self._lost_frames = 0

    @staticmethod
    def _roi_mask(shape, top_frac=0.40, hood_frac=0.06):
        # A wide-FOV low dashcam sees lane lines spread across nearly the
        # full frame width right up until just below the horizon (unlike a
        # narrow-FOV highway dashcam where a trapezoid ROI is the norm), so
        # we just cut out the sky and the car's own hood rather than a
        # narrowing trapezoid, and let slope/position filtering in
        # `_classify_lines` do the real left/right separation.
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        top_y = int(h * top_frac)
        bottom_y = int(h * (1 - hood_frac))
        mask[top_y:bottom_y, :] = 255
        return mask

    @staticmethod
    def _edge_image(frame: np.ndarray) -> np.ndarray:
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        # bright/white-ish lane paint
        bright = cv2.inRange(l_channel, 170, 255)
        # yellow centerline paint
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (15, 80, 120), (40, 255, 255))
        color_mask = cv2.bitwise_or(bright, yellow)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        combined = cv2.bitwise_or(edges, color_mask)
        return combined

    def _classify_lines(self, lines, width):
        left_pts, right_pts = [], []
        if lines is None:
            return left_pts, right_pts
        cx = width / 2
        for line in lines:
            coords = line[0] if hasattr(line[0], "__len__") else line
            x1, y1, x2, y2 = coords
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.25:  # near-horizontal, not a lane edge
                continue
            # a line mostly left of center with negative slope (image coords) -> left lane
            mid_x = (x1 + x2) / 2
            if slope < 0 and mid_x < cx * 1.15:
                left_pts.append((x1, y1))
                left_pts.append((x2, y2))
            elif slope > 0 and mid_x > cx * 0.85:
                right_pts.append((x1, y1))
                right_pts.append((x2, y2))
        return left_pts, right_pts

    @staticmethod
    def _fit_line(points, y_top, y_bottom):
        if len(points) < 2:
            return None
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        if np.all(ys == ys[0]):
            return None
        coeffs = np.polyfit(ys, xs, 1)  # x = m*y + b, robust to vertical lines
        x_top = int(np.polyval(coeffs, y_top))
        x_bottom = int(np.polyval(coeffs, y_bottom))
        max_y_seen = int(ys.max())
        return (x_bottom, y_bottom, x_top, y_top, max_y_seen)

    @staticmethod
    def _x_at_y(line, y):
        x_bottom, y_bottom, x_top, y_top = line[0], line[1], line[2], line[3]
        if y_bottom == y_top:
            return x_bottom
        t = (y - y_top) / (y_bottom - y_top)
        return x_top + t * (x_bottom - x_top)

    def process(self, frame: np.ndarray) -> LaneResult:
        h, w = frame.shape[:2]
        y_bottom = h - 1
        y_top = int(h * 0.45)

        edges = self._edge_image(frame)
        mask = self._roi_mask(frame.shape)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            masked, rho=2, theta=np.pi / 180, threshold=25,
            minLineLength=18, maxLineGap=60,
        )
        left_pts, right_pts = self._classify_lines(lines, w)
        left = self._fit_line(left_pts, y_top, y_bottom) or self._last_left
        right = self._fit_line(right_pts, y_top, y_bottom) or self._last_right

        debug = frame.copy()
        cv2.rectangle(debug, (0, y_top), (w - 1, y_bottom), (80, 80, 80), 1, cv2.LINE_AA)

        confidence = 0.0
        if left is not None and right is not None:
            cv2.line(debug, (left[0], left[1]), (left[2], left[3]), (255, 80, 0), 4, cv2.LINE_AA)
            cv2.line(debug, (right[0], right[1]), (right[2], right[3]), (0, 165, 255), 4, cv2.LINE_AA)

            # Evaluate the offset at a row close to where both lines
            # actually had detected pixels, not at the frame's bottom edge:
            # a short Hough segment found only near the horizon, linearly
            # extrapolated all the way down, can overshoot wildly on a
            # sharp bend (one line often exits the frame sideways before
            # reaching the bottom row). Capping the eval row at the
            # shallower of the two lines' real data keeps the estimate
            # grounded in what was actually observed.
            eval_y = min(y_bottom, max(left[4], right[4]), int(h * 0.88))
            eval_y = max(eval_y, y_top + 1)
            left_x = self._x_at_y(left, eval_y)
            right_x = self._x_at_y(right, eval_y)

            lane_center = (left_x + right_x) / 2
            lane_width = max(1.0, right_x - left_x)

            offset_px = (w / 2) - lane_center
            offset_norm = float(np.clip(offset_px / (lane_width / 2), -2.5, 2.5))
            # curvature proxy: how much the lane center shifts between the
            # eval row and the top of the ROI
            lane_center_top = (left[2] + right[2]) / 2
            curvature = float((lane_center_top - lane_center) / max(1.0, (eval_y - y_top)))

            self._lost_frames = 0
            confidence = 1.0
            self._last_left, self._last_right = left, right
        else:
            self._lost_frames += 1
            confidence = max(0.0, 0.6 - 0.12 * self._lost_frames)
            offset_px = self._offset_hist[-1] if self._offset_hist else 0.0
            offset_norm = offset_px
            curvature = self._curve_hist[-1] if self._curve_hist else 0.0

        self._offset_hist.append(offset_norm)
        self._curve_hist.append(curvature)
        smoothed_offset = float(np.mean(self._offset_hist))
        smoothed_curve = float(np.mean(self._curve_hist))

        cv2.circle(debug, (int(w / 2), y_bottom - 10), 5, (255, 255, 255), -1)
        if left is not None and right is not None:
            cv2.circle(debug, (int(lane_center), y_bottom - 10), 5, (0, 255, 0), -1)
        cv2.putText(debug, f"offset={smoothed_offset:+.2f}  curve={smoothed_curve:+.3f}  conf={confidence:.2f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return LaneResult(
            offset_px=offset_px,
            offset_norm=smoothed_offset,
            curvature=smoothed_curve,
            confidence=confidence,
            left_line=left,
            right_line=right,
            debug=debug,
        )
