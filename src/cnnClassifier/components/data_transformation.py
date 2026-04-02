import os
import shutil
import pandas as pd
from pathlib import Path
from cnnClassifier import logger

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def organize_dataset(self, csv_file, image_folder):
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            img_id = row['id_code']
            label = str(row['diagnosis'])

            possible_paths = [
                os.path.join(image_folder, img_id + ".png"),
                os.path.join(image_folder, img_id + ".jpg")
            ]

            src = None
            for path in possible_paths:
                if os.path.exists(path):
                    src = path
                    break

            if src is None:
                print(f"❌ Not found: {img_id}")
                continue

            dst_folder = os.path.join(self.config.transformed_data_path, label)
            os.makedirs(dst_folder, exist_ok=True)

            shutil.copy(src, os.path.join(dst_folder, os.path.basename(src)))

    def run(self):
        base_path = self.config.base_data_path

        self.organize_dataset(
            os.path.join(base_path, "train_1.csv"),
            os.path.join(base_path, "train_images", "train_images")
        )

        self.organize_dataset(
            os.path.join(base_path, "test.csv"),
            os.path.join(base_path, "test_images", "test_images")
        )

        self.organize_dataset(
            os.path.join(base_path, "valid.csv"),
            os.path.join(base_path, "val_images", "val_images")
        )

        logger.info(f"✅ Data Transformation Completed")