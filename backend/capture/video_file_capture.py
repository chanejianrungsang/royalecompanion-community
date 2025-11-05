from __future__ import annotations

import cv2


class VideoFileCapture:
    """Simple video file reader that mimics the capture_window interface."""

    def __init__(self, path: str, target_size: tuple[int, int] | None = (900, 1600)):
        self.path = path
        self._cap = cv2.VideoCapture(path)
        self.target_size = target_size
        self.width = None
        self.height = None
        self.fps: float | None = None
        self.frame_interval_ms: int | None = None

        if self._cap and self._cap.isOpened():
            native_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            native_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps > 0:
                self.fps = fps
                self.frame_interval_ms = max(1, int(round(1000.0 / fps)))
            else:
                self.fps = None
                self.frame_interval_ms = None
            if target_size:
                self.width, self.height = target_size
            elif native_width and native_height:
                self.width, self.height = native_width, native_height
        else:
            self.fps = None
            self.frame_interval_ms = None

    @property
    def is_open(self) -> bool:
        return bool(self._cap and self._cap.isOpened())

    def capture_window(self):
        if not self.is_open:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        # Normalize to portrait orientation (height > width).
        if frame.shape[1] > frame.shape[0]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if self.target_size is not None:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            self.width, self.height = self.target_size
        else:
            self.height, self.width = frame.shape[:2]

        return frame

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None
