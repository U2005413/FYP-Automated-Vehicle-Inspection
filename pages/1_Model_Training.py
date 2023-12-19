import shutil
import time

import streamlit as st
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

from utils.config_manager import write_config_file
from utils.file_manager import get_filename_wo_ext
from utils.incremental_learning import incremental_learning


class Model_Training_Page:
    def __init__(self):
        self.model = st.session_state["model"]
        self.config = st.session_state["config"]
        if "key" not in st.session_state:
            st.session_state["key"] = 1

    def save_images(self, images, folder_path, name_size_map):
        for image in images:
            im = Image.open(image)
            im.save(folder_path + "\\" + image.name)

            name_size_map[get_filename_wo_ext(image.name)] = im.size

    def save_text_files(self, files, folder_path, name_size_map):
        index = len(self.model.names)

        for file in files:
            coords = list(map(int, str(file.read(), "utf-8").split(" ")))
            coords = self.normalize_box(
                coords, name_size_map[get_filename_wo_ext(file.name)]
            )

            content = f"{index} {coords[0]} {coords[1]} {coords[2]} {coords[3]}"
            file_path = folder_path + "\\" + file.name
            with open(file_path, "w") as f:
                f.write(content)

    def reset_page(self):
        st.session_state["key"] = (st.session_state["key"] + 1) % 2
        st.rerun()

    def save_all_uploaded_files(
        self, train_images, val_images, train_labels, val_labels
    ):
        name_size_map = {}

        self.save_images(train_images, "datasets\\images\\train", name_size_map)
        self.save_images(val_images, "datasets\\images\\val", name_size_map)
        self.save_text_files(train_labels, "datasets\\labels\\train", name_size_map)
        self.save_text_files(val_labels, "datasets\\labels\\val", name_size_map)

    def write_yaml(self, labels_dict):
        yaml_data = [
            {"path": "."},
            {"train": "images\\train"},
            {"val": "images\\val"},
            {"names": labels_dict},
        ]

        with open("dataset.yaml", "w") as f:
            for data in yaml_data:
                yaml.dump(data, f)

    def create_dataset_yaml(self, label):
        index_label_map = self.model.names.copy()
        index = len(index_label_map)
        index_label_map[index] = label

        self.write_yaml(index_label_map)

    def normalize_box(self, coords, size):
        width = size[0]
        height = size[1]

        x1 = coords[0] / width
        y1 = coords[1] / height
        x2 = coords[2] / width
        y2 = coords[3] / height

        return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]

    def display_success(self, save_dir):
        success = st.success("Training Results saved to " + save_dir)
        time.sleep(3)
        success.empty()

    def save_model_weight(self, result_dir):
        destination = (
            self.config["models_path"]
            + "\\"
            + str(len(self.model.names))
            + "_Classes.pt"
        )
        shutil.copy(result_dir + "\\weights\\best.pt", destination)

        self.config["model"] = destination
        write_config_file(self.config)

    def split_images_and_labels(self, images, labels):
        splitted_images = train_test_split(images, test_size=0.2)
        train_images = splitted_images[0]
        val_images = splitted_images[1]

        train_image_names = []
        for img in train_images:
            name = get_filename_wo_ext(img.name) + ".txt"
            train_image_names.append(name)

        train_labels = []
        val_labels = []
        for label in labels:
            if label.name not in train_image_names:
                val_labels.append(label)
            else:
                train_labels.append(label)
        return train_images, val_images, train_labels, val_labels

    def run(self):
        st.set_page_config(
            page_title="Vehicle Inspection System",
            page_icon="🛠️",
        )

        st.title("Model Training Page")

        images = st.file_uploader(
            "Images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key=st.session_state["key"],
        )

        labels = st.file_uploader(
            "Labels",
            type=["txt"],
            accept_multiple_files=True,
            key=st.session_state["key"] + 1,
        )

        label_name = st.text_input(
            "New Car Model Name",
            placeholder="Myvi 2005",
            key=st.session_state["key"] + 4,
        )
        if label_name:
            upload_button = st.button("Upload")
            if upload_button:
                (
                    train_images,
                    val_images,
                    train_labels,
                    val_labels,
                ) = self.split_images_and_labels(images, labels)

                self.save_all_uploaded_files(
                    train_images, val_images, train_labels, val_labels
                )

                self.create_dataset_yaml(label_name)
                self.reset_page()

        training_epochs = st.number_input(
            "Training Epochs (Upload before incremental learning)",
            value=50,
            min_value=2,
        )
        if st.button("Train Model"):
            with st.spinner("Model Training In Progress..."):
                save_dir = incremental_learning(
                    self.model, "dataset.yaml", training_epochs
                )

                self.save_model_weight(save_dir)

            self.display_success(save_dir)


Model_Training_Page().run()
