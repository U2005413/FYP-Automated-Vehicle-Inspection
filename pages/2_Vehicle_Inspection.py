import os

import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer

from utils.video import VideoProcessor


class Inspection_Page:
    def __init__(self):
        self.config = st.session_state["config"]
        self.model = st.session_state["model"]
        self.heatmap_model = st.session_state["heatmap_model"]
        self.original_screenshots_path = self.config["original_screenshots_path"]
        self.cam = None

    def screenshot_button_func(self):
        with self.cam.video_processor.frame_lock:
            current_screenshot = self.cam.video_processor.in_image
            st.image(current_screenshot)

            screenshot_count = len(os.listdir(self.original_screenshots_path)) + 1
            screenshot_name = "Screenshot" + str(screenshot_count) + ".jpg"

            st.button(
                "Save",
                on_click=lambda: self.save_screenshot(
                    current_screenshot, screenshot_name
                ),
            )

    def save_screenshot(self, current_screenshot, file_name):
        im = Image.fromarray(current_screenshot)
        im.save(self.original_screenshots_path + "\\" + file_name)

    def run(self):
        st.set_page_config(
            page_title="Vehicle Inspection System",
            page_icon="🛠️",
        )

        st.title("Vehicle Inspection Page")

        toggle = st.toggle("Activate Heatmap")
        if toggle:
            self.cam = webrtc_streamer(
                key="camera",
                video_processor_factory=lambda: VideoProcessor(
                    self.model, self.heatmap_model
                ),
            )
        else:
            self.cam = webrtc_streamer(
                key="camera", video_processor_factory=lambda: VideoProcessor(self.model)
            )

        if self.cam.video_processor:
            if st.button("Screenshot"):
                self.screenshot_button_func()


Inspection_Page().run()
