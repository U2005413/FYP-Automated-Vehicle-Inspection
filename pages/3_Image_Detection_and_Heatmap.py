import os

import cv2
import streamlit as st

from utils.file_manager import *


class Detection_Page:
    def __init__(self):
        self.config = st.session_state["config"]
        self.model = st.session_state["model"]
        self.heatmap_model = st.session_state["heatmap_model"]
        self.original_screenshots_path = self.config["original_screenshots_path"]
        self.heatmaps_path = self.config["heatmaps_path"]
        self.detections_path = self.config["detections_path"]

    def apply_detection(self, img_name):
        img_path = self.original_screenshots_path + "\\" + img_name
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.model.predict(img, verbose=False)[0].plot()

        save_image_arr(
            result,
            self.detections_path
            + "\\"
            + get_filename_wo_ext(img_name)
            + "_detection.jpg",
        )

    def apply_gradcam(self, img_name):
        img_path = self.original_screenshots_path + "\\" + img_name
        result = self.heatmap_model(img_path=img_path)

        save_image_arr(
            result,
            self.heatmaps_path + "\\" + get_filename_wo_ext(img_name) + "_heatmap.jpg",
        )

    def apply_image_mask(self, selected_images, apply_func):
        for image_path in selected_images:
            apply_func(image_path)

    def run(self):
        st.set_page_config(
            page_title="Vehicle Inspection System",
            page_icon="🛠️",
        )

        st.title("Screenshots Page")

        image_paths = os.listdir(self.original_screenshots_path)
        selected_images = st.multiselect(label="Images", options=image_paths)

        if selected_images:
            detection_button = st.button("Apply Detection")
            if detection_button:
                self.apply_image_mask(selected_images, self.apply_detection)

            heatmap_button = st.button("Apply Heatmap")
            if heatmap_button:
                self.apply_image_mask(selected_images, self.apply_gradcam)

            ori_tab, detection_tab, heatmap_tab = st.tabs(
                ["Original", "Detection", "Heatmap"]
            )

            self.show_original_images(selected_images, ori_tab)
            self.show_tab(selected_images, detection_tab, self.detections_path)
            self.show_tab(selected_images, heatmap_tab, self.heatmaps_path)

    def show_original_images(self, selected_images, ori_tab):
        for i, image_path in enumerate(selected_images):
            if i % 3 == 0:
                cols = ori_tab.columns(3)
            cols[i % 3].image(self.original_screenshots_path + "\\" + image_path)

    def show_tab(self, selected_images, tab, tab_path):
        for i, image_path in enumerate(selected_images):
            if i % 3 == 0:
                cols = tab.columns(3)

            filepath = self.get_processed_file(image_path, tab_path)
            if os.path.exists(filepath):
                cols[i % 3].image(filepath)

    def get_processed_file(self, original_image_path, tab_path):
        if tab_path == self.config["detections_path"]:
            ext = "_detection.jpg"
        else:
            ext = "_heatmap.jpg"
        return tab_path + "\\" + get_filename_wo_ext(original_image_path) + ext


Detection_Page().run()
