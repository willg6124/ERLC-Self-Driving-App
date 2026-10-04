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
    # Only a backstop against total garbage (e.g. the old extrapolation-
    # overshoot bug that produced a "lane" ~900px wide out of a 960px
    # frame). Perspective means the exact same real lane can legitimately
    # read anywhere from a couple hundred px (far eval row, near horizon)
    # to most of the frame width (close eval row, right at the car) - a
    # tight fixed ceiling just flaps on/off whenever a frame's eval row
    # happens to land a few px either side of some arbitrary threshold
    # even though the detection itself is rock steady (confirmed: width
    # sitting at 850-855px against a hand-picked ~851px ceiling flickered
    # confidence between 1.0 and 0.0 every other frame for a perfectly
    # consistent detection). Real implausibility detection is relative:
    # compare this frame's width to the recent rolling median instead.
    MIN_LANE_PX = 60
    MAX_LANE_PX = 1500
    WIDTH_HISTORY = 15
    MAX_RELATIVE_WIDTH_JUMP = 0.4  # 40% off the recent median -> suspect

    def __init__(self, smoothing: int = 7):
        self._offset_hist = deque(maxlen=smoothing)
        self._curve_hist = deque(maxlen=smoothing)
        self._width_hist = deque(maxlen=self.WIDTH_HISTORY)
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

    def _classify_lines(self, lines, width, y_bottom):
        """Split raw Hough segments into left/right lane-edge candidates.

        Classifying by a segment's raw midpoint-vs-center (the original
        approach) uses a deliberately fuzzy +/-15% overlap band around the
        frame center so that lines are not lost on sharp bends. But that
        same fuzziness means two segments a few pixels apart near dead
        center can land on *opposite* sides of the split, which silently
        swaps which physical lane edge feeds the left vs right fit and
        made `offset_norm` discontinuously flip sign when the car was
        near the lane center (verified: a 0.5m lateral nudge near center
        flipped the smoothed offset from +0.74 to -0.93). Instead,
        extrapolate each segment's *own* slope out to the bottom row (the
        row nearest the car, i.e. the most geometrically meaningful place
        to ask "which side of the car is this edge on") and classify by
        that unambiguous bottom-anchored x with no overlap band.
        """
        left_segs, right_segs = [], []
        if lines is None:
            return left_segs, right_segs
        cx = width / 2
        for line in lines:
            coords = line[0] if hasattr(line[0], "__len__") else line
            x1, y1, x2, y2 = coords
            if x2 == x1 or y2 == y1:
                continue
            slope_dydx = (y2 - y1) / (x2 - x1)
            if abs(slope_dydx) < 0.25:  # near-horizontal, not a lane edge
                continue
            # extrapolate to the bottom row using x = m*y + b fit on this
            # single segment, so classification reflects where the edge
            # actually sits relative to the car, not an arbitrary midpoint.
            m = (x2 - x1) / (y2 - y1)
            x_at_bottom = x1 + m * (y_bottom - y1)
            if slope_dydx < 0 and x_at_bottom < cx:
                left_segs.append(((x1, y1), (x2, y2), x_at_bottom))
            elif slope_dydx > 0 and x_at_bottom > cx:
                right_segs.append(((x1, y1), (x2, y2), x_at_bottom))
        return self._select_innermost_cluster(left_segs, cx), self._select_innermost_cluster(right_segs, cx)

    @staticmethod
    def _select_innermost_cluster(segs, cx, tolerance_px=70.0):
        """A wide, bright render also draws the road's own outer
        shoulder/grass border in the same near-white threshold as real
        lane paint, so `left_segs`/`right_segs` can contain segments from
        TWO physically different lines (the true near lane edge AND the
        far-away road border). Lumping both into one polyfit produces a
        lane width far outside anything real (e.g. ~850px when a true
        in-lane pair is ~300-400px here), which made `offset_norm` jump
        around badly. By construction the TRUE adjacent lane edge is
        always the innermost (closest-to-center) candidate on each side -
        any other matching line out there is some other lane/road-edge
        marking further away - so cluster segments by their bottom-row x
        and keep only the points belonging to the cluster nearest cx.
        """
        if not segs:
            return []
        segs = sorted(segs, key=lambda s: abs(s[2] - cx))
        anchor_x = segs[0][2]
        kept = [s for s in segs if abs(s[2] - anchor_x) <= tolerance_px]
        points = []
        for p1, p2, _ in kept:
            points.append(p1)
            points.append(p2)
        return points

    @staticmethod
    def _reject_outliers(points, y_bottom, y_top):
        """Drop points whose implied bottom-row x is far from the group's
        median before fitting, so a handful of spurious Hough segments
        (opposite lane, shoulder marking, a clipped frame-edge artifact)
        can't drag the whole polynomial fit off to one side."""
        if len(points) < 4:
            return points
        ys = np.array([p[1] for p in points], dtype=float)
        xs = np.array([p[0] for p in points], dtype=float)
        if np.all(ys == ys[0]):
            return points
        # rough single-pass linear fit just to project every point to a
        # common reference row for outlier comparison
        coeffs = np.polyfit(ys, xs, 1)
        resid = xs - np.polyval(coeffs, ys)
        med = np.median(resid)
        mad = np.median(np.abs(resid - med)) + 1e-6
        keep = np.abs(resid - med) < max(25.0, 6.0 * mad)
        kept = [p for p, k in zip(points, keep) if k]
        return kept if len(kept) >= 2 else points

    # This wide low-FOV dashcam view genuinely produces lane-edge slopes
    # around dx/dy ~= 3 for a close, near-centered edge (verified against
    # real Hough segments), so this is only a backstop against truly
    # degenerate fits (e.g. a point cluster with almost no y spread,
    # where polyfit's slope blows up toward vertical-in-x nonsense) -
    # NOT a plausibility filter on normal lane geometry. The real defense
    # against extrapolation blowup is capping eval_y to each line's own
    # observed y-range below, not rejecting on slope.
    MAX_PLAUSIBLE_SLOPE = 8.0

    @staticmethod
    def _fit_line(points, y_top, y_bottom):
        if len(points) < 2:
            return None
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        if np.all(ys == ys[0]):
            return None
        coeffs = np.polyfit(ys, xs, 1)  # x = m*y + b, robust to vertical lines
        if abs(coeffs[0]) > LaneDetector.MAX_PLAUSIBLE_SLOPE:
            return None
        x_top = int(np.polyval(coeffs, y_top))
        x_bottom = int(np.polyval(coeffs, y_bottom))
        max_y_seen = int(ys.max())
        min_y_seen = int(ys.min())
        return (x_bottom, y_bottom, x_top, y_top, max_y_seen, min_y_seen)

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
        left_pts, right_pts = self._classify_lines(lines, w, y_bottom)
        left_pts = self._reject_outliers(left_pts, y_bottom, y_top)
        right_pts = self._reject_outliers(right_pts, y_bottom, y_top)
        left = self._fit_line(left_pts, y_top, y_bottom) or self._last_left
        right = self._fit_line(right_pts, y_top, y_bottom) or self._last_right

        debug = frame.copy()
        cv2.rectangle(debug, (0, y_top), (w - 1, y_bottom), (80, 80, 80), 1, cv2.LINE_AA)

        confidence = 0.0
        width_implausible = False
        if left is not None and right is not None:
            cv2.line(debug, (left[0], left[1]), (left[2], left[3]), (255, 80, 0), 4, cv2.LINE_AA)
            cv2.line(debug, (right[0], right[1]), (right[2], right[3]), (0, 165, 255), 4, cv2.LINE_AA)

            # Evaluate the offset at a row inside the range BOTH lines
            # actually had detected pixels, not at the frame's bottom
            # edge: a short Hough segment found only near the horizon,
            # linearly extrapolated all the way down, can overshoot
            # wildly (one line often has data only near the horizon while
            # the other reaches much closer to the car). The eval row
            # must be capped at the SHALLOWER of the two lines' max
            # observed y (min(...), not max(...) - using max here was a
            # bug that let the shallow-data line get extrapolated far
            # past its own observed pixels, producing physically
            # impossible lane centers / widths and a wildly unstable
            # offset) and floored at the deeper of their min observed y.
            hi = min(y_bottom, min(left[4], right[4]), int(h * 0.88))
            eval_y = max(hi, y_top + 1)
            left_x = self._x_at_y(left, eval_y)
            right_x = self._x_at_y(right, eval_y)

            lane_center = (left_x + right_x) / 2
            lane_width = max(1.0, right_x - left_x)

            # Sanity-check the detected lane width: absolute bounds only
            # catch total garbage (crossed left/right, a lock onto an
            # unrelated edge far away). Genuine instability -- the width
            # suddenly jumping relative to what this detector has been
            # consistently reporting -- is caught by comparing to the
            # recent rolling median instead of a fixed px ceiling, so a
            # perfectly steady reading near some arbitrary threshold
            # doesn't flap confidence on and off every other frame.
            width_implausible = not (self.MIN_LANE_PX <= lane_width <= self.MAX_LANE_PX)
            if not width_implausible and len(self._width_hist) >= 4:
                med = float(np.median(self._width_hist))
                if med > 1.0 and abs(lane_width - med) / med > self.MAX_RELATIVE_WIDTH_JUMP:
                    width_implausible = True

            offset_px = (w / 2) - lane_center
            offset_norm = float(np.clip(offset_px / (lane_width / 2), -2.5, 2.5))
            # curvature proxy: how much the lane center shifts between the
            # eval row and the top of the ROI
            lane_center_top = (left[2] + right[2]) / 2
            curvature = float((lane_center_top - lane_center) / max(1.0, (eval_y - y_top)))

            if width_implausible:
                self._lost_frames += 1
                confidence = max(0.0, 0.5 - 0.12 * self._lost_frames)
                if self._offset_hist:
                    offset_norm = self._offset_hist[-1]
                    curvature = self._curve_hist[-1]
                # don't let a corrupted fit poison next frame's fallback
            else:
                self._lost_frames = 0
                confidence = 1.0
                self._last_left, self._last_right = left, right
                self._width_hist.append(lane_width)
        else:
            self._lost_frames += 1
            confidence = max(0.0, 0.6 - 0.12 * self._lost_frames)
            offset_px = self._offset_hist[-1] if self._offset_hist else 0.0
            offset_norm = offset_px
            curvature = self._curve_hist[-1] if self._curve_hist else 0.0

        # Jump-continuity gate: a real lane-center offset can't teleport
        # between consecutive frames at normal frame rates. If a
        # (nominally high-confidence) single-frame reading implies an
        # implausibly large jump from the recent smoothed value, it's far
        # more likely a left/right mix-up or a bad outlier fit than a real
        # instantaneous lane change, so clamp the step and knock down
        # confidence instead of letting it whip the steering controller.
        MAX_STEP = 0.6
        if len(self._offset_hist) >= 3:
            prev = float(np.mean(self._offset_hist))
            delta = offset_norm - prev
            if abs(delta) > MAX_STEP:
                offset_norm = prev + MAX_STEP * np.sign(delta)
                confidence = min(confidence, 0.5)

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
