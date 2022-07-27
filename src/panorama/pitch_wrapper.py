import cv2
import numpy as np
from timeit import default_timer as time
from stitching_utils.old_version.stitching import plot_image, find_features_, match_features_, create_pano, filter_frames
from stitching_utils.extraction import  Video
from stitching_utils.img_stitch import ImageStitcher


class PitchWrapper(object):
    """Pitch building"""
    def __init__(self, initial_frames: list):
        """
        Class constructor
        Args:
            initial_frames:
        """
        extractor = Video()
        pano_frames = extractor.extract_keyframes(20, initial_frames)
        # frames = np.array(initial_frames)
        # thresh = find_threshold(frames)
        # key_indices = key_frames_(frames, thresh)
        # pano_frames = frames[key_indices]
        stitch = ImageStitcher(pano_frames)
        self.pano = stitch.stitch()
        #pano_frames, points_descriptors = filter_frames(pano_frames)
        #self.pano = create_pano(pano_frames, points_descriptors)
        pitch_area = cv2.cvtColor(self.pano, cv2.COLOR_RGB2GRAY)
        _, pitch_area = cv2.threshold(pitch_area, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(pitch_area, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

        self.viz_shift = 0 # pitch_area.shape[1] // 10

        pitch_area = np.zeros(
            (pitch_area.shape[0] + self.viz_shift,
             pitch_area.shape[1] + self.viz_shift * 10, 3),
            dtype=np.uint8
        )

        if len(contours) > 0:
            contours.sort(
                key=lambda cnt: cv2.arcLength(cnt, True), reverse=True
            )
            contour = contours[0]
            contour = cv2.convexHull(contour)

            self.pitch_scheme = cv2.drawContours(
                pitch_area,
                [np.array(contour) + [self.viz_shift, self.viz_shift]],
                -1, (20, 200, 0), 25
            )
        else:
            self.pitch_scheme = pitch_area

        self.pano_kp, self.pano_des = find_features_(self.pano)

    def size(self) -> tuple:
        """
        Size of built pano
        Returns:
            Return tuple with size: (Height, Width)
        """
        return self.pano.shape[0], self.pano.shape[1]

    def estimate_matrix(self, _frame: np.ndarray):
        """
        Estimate warp matrix
        Args:
            _frame: image in RGB HWC uint8 format

        Returns:
            Perspective matrix
        """
        k = 560 / max(_frame.shape)
        kp1, des1 = find_features_(
            cv2.resize(_frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        )

        kp1 = kp1 / k

        matches = match_features_(des1, self.pano_des, .7)
        src_pts = np.float32([kp1[m[0]] for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [self.pano_kp[m[1]] for m in matches]
        ).reshape(-1, 1, 2)

        m_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

        return m_matrix

    def __call__(self,
                 _frame: np.ndarray,
                 scene_points: np.ndarray) -> np.ndarray:
        """
        Wrap frame scene points to global coordinates
        Args:
            _frame: image in RGB HWC uint8 format
            scene_points: array of points in BXY format

        Returns:
            Array of points in BXY format
        """
        m_matrix = self.estimate_matrix(_frame)

        warped_frame_points = cv2.perspectiveTransform(
            np.expand_dims(scene_points, 1).astype(np.float32), m_matrix)
        warped_frame_points = warped_frame_points.squeeze(1).astype(np.int32)

        return warped_frame_points
