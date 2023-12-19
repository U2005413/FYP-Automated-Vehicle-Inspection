import os

import streamlit as st
from ultralytics import YOLO

from utils.config_manager import *
from utils.grad_cam import yolov8_heatmap


class Home_Page:
    def __init__(self):
        self.config = self.load_config()
        self.model = self.load_model()
        self.heatmap_model = self.load_heatmap_model()

    def display_error(self, invalids):
        body = "The following directory(s) is/are invalid: \n\n"
        for invalid in invalids:
            body += invalid + "\n\n"

        st.error(icon="🚨", body=body)

    def load_config(self):
        if "config" not in st.session_state:
            config = load_config_file()
            st.session_state["config"] = config
        return st.session_state["config"]

    def load_model(self):
        if "model" not in st.session_state:
            model = YOLO(self.config["model"])
            st.session_state["model"] = model
        return st.session_state["model"]

    def load_heatmap_model(self):
        if "heatmap_model" not in st.session_state:
            model = yolov8_heatmap(**self.get_heatmap_params())
            st.session_state["heatmap_model"] = model
        return st.session_state["heatmap_model"]

    def get_heatmap_params(self):
        params = {
            "weight": self.config["model"],
            "cfg": "yolov8n.yaml",
            "device": "cpu",
            "layer": "model.model[-2]",
            "backward_type": "all",
        }
        return params

    def validate_directories(self, properties_map):
        invalids = []
        for property in properties_map:
            if not os.path.exists(properties_map[property]):
                invalids.append(properties_map[property])
        return invalids

    def save_button_func(self, properties_map):
        invalids = self.validate_directories(properties_map)
        if len(invalids) > 0:
            self.display_error(invalids)
            return

        for property in properties_map:
            self.config[property] = properties_map[property]
        write_config_file(self.config)

        st.session_state.pop("config")
        st.session_state.pop("model")
        st.session_state.pop("heatmap_model")
        st.rerun()

    def run(self):
        st.set_page_config(
            page_title="Vehicle Inspection System",
            page_icon="🛠️",
        )

        st.title("Vehicle Inspection System 🛠️")

        models_path = st.text_input(
            "Models Saving Folder", value=self.config["models_path"]
        )
        model = st.text_input("Detection Model", value=self.config["model"])
        original_screenshots_path = st.text_input(
            "Screenshots Saving Folder", value=self.config["original_screenshots_path"]
        )
        detections_path = st.text_input(
            "Detections Saving Folder", value=self.config["detections_path"]
        )
        heatmaps_path = st.text_input(
            "Heatmaps Saving Folder", value=self.config["heatmaps_path"]
        )

        properties_map = {
            "models_path": models_path,
            "model": model,
            "original_screenshots_path": original_screenshots_path,
            "detections_path": detections_path,
            "heatmaps_path": heatmaps_path,
        }
        if st.button("Save"):
            self.save_button_func(properties_map)


Home_Page().run()
