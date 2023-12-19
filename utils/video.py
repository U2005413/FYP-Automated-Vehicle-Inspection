import threading

import av
from streamlit_webrtc import VideoProcessorBase


class VideoProcessor(VideoProcessorBase):
    def __init__(self, model, heatmap_model=None):
        self.model = model
        self.heatmap_model = heatmap_model
        self.frame_lock = threading.Lock()
        self.in_image = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        in_image = frame.to_ndarray(format="rgb24")

        with self.frame_lock:
            self.in_image = in_image

        out_image = self.detect(in_image)[0].plot()
        if self.heatmap_model:
            out_image = self.heatmap_model(img=out_image)

        return av.VideoFrame.from_ndarray(out_image, format="rgb24")

    def detect(self, image):
        return self.model.predict(image, verbose=False)

    def set_show_heatmap(self, show_heatmap):
        self.show_heatmap = show_heatmap
