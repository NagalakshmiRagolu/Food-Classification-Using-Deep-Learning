'''
food classification using the convolutional neural networks
'''

import warnings
warnings.filterwarnings('ignore')


import contextlib
import io
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import sys
import random
import json
import shutil
import h5py
from PIL import Image
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import logging
from log_code import setup_logging
logger = setup_logging('main1')


class FOOD_CLASSIFICATION:
    def __init__(self,path):
        try:
            self.path = path
            # keep logger accessible as self.logger so all new methods use self.logger.info(...)
            self.logger = logger
            self.logger.info(f"FOOD_CLASSIFICATION instance initialized with dataset path: {self.path}")

        except Exception as e:
            e_type, e_value, e_traceback = sys.exc_info()
            logger.error(f"Error in __init__ on line {e_traceback.tb_lineno} because {e_value}")

    def check_classnames(self):
        try:
            class_names = []
            for name in os.listdir(self.path):
                if os.path.isdir(os.path.join(self.path, name)):
                    class_names.append(name)

            self.logger.info(f"class names are : {class_names}")

            self.nutri_info = []
            for food_name in class_names:
                info = {
                    "food_name": food_name,
                    "nutritional_info": {
                        "protein": f"{random.randint(1, 30)}g",
                        "fiber": f"{random.randint(1, 10)}g",
                        "calories": random.randint(100, 600),
                        "carbohydrates": f"{random.randint(10, 70)}g",
                        "fat": f"{random.randint(1, 25)}g"
                    }
                }
                self.nutri_info.append(info)

            self.logger.info(f"nutritional info generated for {len(self.nutri_info)} items")
            return self.nutri_info

        except Exception as e:
            e_type, e_value, e_traceback = sys.exc_info()
            self.logger.error(f"Error in check_classnames on line {e_traceback.tb_lineno} because {e_value}")

    def json_data(self):
        try:
            data = self.check_classnames()
            self.logger.info(f"Data received in json_data(): {len(data)} items")
            json_path = r'C:\INTERNSHIP_VIHARATECH\INTERN_PR2\food_nutrition.json'
            with open(json_path, 'w') as json_file:
                json.dump(data,json_file,indent=4)

            self.logger.info(f"JSON file created successfully at: {json_path}")

        except Exception as e:
            e_type, e_value, e_traceback = sys.exc_info()
            self.logger.error(f"Error in json_data on line {e_traceback.tb_lineno} because {e_value}")

    def selected_images(self):
        try:
            output_path = r'C:\INTERNSHIP_VIHARATECH\INTERN_PR2\Selected_images'
            images_per_class = 200
            if not os.path.exists(output_path):
                os.makedirs(output_path)
                self.logger.info(f"Created output folder : {output_path}")
                for class_name in os.listdir(self.path):
                    class_dir = os.path.join(self.path,class_name)
                    if not os.path.isdir(class_dir):
                        continue
        # create output sub folder for each class
                    output_class_dir = os.path.join(output_path, class_name)
                    os.makedirs(output_class_dir, exist_ok=True)

             # List all images
                    images =[]
                    for img in os.listdir(class_dir):
                        if img.lower().endswith(('.jpg', '.jpeg')):
                            images.append(img)
                    self.logger.info(f"{class_name}: found {len(images)} images")

            # Randomly select 200 (or fewer)
                    selected_images = random.sample(images, min(images_per_class, len(images)))

            # Copy selected images
                    for img in selected_images:
                        src = os.path.join(class_dir, img)
                        dst = os.path.join(output_class_dir, img)
                        shutil.copy2(src, dst)

                    self.logger.info(f"Copied {len(selected_images)} images from {class_name}")

        except Exception as e:
            e_type, e_value, e_traceback = sys.exc_info()
            self.logger.error(f"Error in selected_images on line {e_traceback.tb_lineno} because {e_value}")

    def splitting_data(self):
        try:
            self.df = r'C:\INTERNSHIP_VIHARATECH\INTERN_PR2\Selected_images'
            self.base_output = r'C:\INTERNSHIP_VIHARATECH\INTERN_PR2\Split_data'
            self.train_ratio = 0.6
            self.valid_ratio = 0.2
            self.test_ratio = 0.2

            self.train_dir = os.path.join(self.base_output, 'Train')
            self.valid_dir = os.path.join(self.base_output, 'Validation')
            self.test_dir = os.path.join(self.base_output, 'Test')

            for folder in [self.train_dir, self.valid_dir, self.test_dir]:
                os.makedirs(folder, exist_ok=True)

            # Image resize
            self.img_resize = (220, 220)
            for class_name in os.listdir(self.df):
                self.class_folder = os.path.join(self.df, class_name)
                if not os.path.isdir(self.class_folder):
                    continue

                # Create class folders inside each split BEFORE processing images
                os.makedirs(os.path.join(self.train_dir, class_name), exist_ok=True)
                os.makedirs(os.path.join(self.valid_dir, class_name), exist_ok=True)
                os.makedirs(os.path.join(self.test_dir, class_name), exist_ok=True)

                self.re_images = []
                for f in os.listdir(self.class_folder):
                    if f.lower().endswith(('.jpg', '.jpeg')):
                        self.re_images.append(f)

                # Resize images before splitting
                for img_name in self.re_images:
                    src_path = os.path.join(self.class_folder, img_name)
                    img = Image.open(src_path).convert('RGB')
                    img = img.resize(self.img_resize)
                    img.save(src_path, format='JPEG', quality=95)

                random.shuffle(self.re_images)

                self.total = len(self.re_images)
                self.train_len = int(self.total * self.train_ratio)
                self.valid_len = int(self.total * self.valid_ratio)
                self.test_len = self.total - self.train_len - self.valid_len

                self.train_images = self.re_images[:self.train_len]
                self.val_images = self.re_images[self.train_len:self.train_len + self.valid_len]
                self.test_images = self.re_images[self.train_len + self.valid_len:]

                # Copy images to split directories
                for img_name in self.train_images:
                    shutil.copy(os.path.join(self.class_folder, img_name),
                                os.path.join(self.train_dir, class_name, img_name))
                for img_name in self.val_images:
                    shutil.copy(os.path.join(self.class_folder, img_name),
                                os.path.join(self.valid_dir, class_name, img_name))
                for img_name in self.test_images:
                    shutil.copy(os.path.join(self.class_folder, img_name),
                                os.path.join(self.test_dir, class_name, img_name))

                self.logger.info(
                    f"{class_name}: Train={len(self.train_images)}, Valid={len(self.val_images)}, Test={len(self.test_images)}")

            self.logger.info(f'_______Data splitting is successfully completed________')

        except Exception as e:
            e_type, e_value, e_traceback = sys.exc_info()
            self.logger.error(f"Error in splitting_data on line {e_traceback.tb_lineno} because {e_value}")

    def data_grouping(self):
        try:
            # Log dataset lengths
            self.logger.info(f'checking length of train data : {len(self.train_images)}')
            self.logger.info(f'checking length of validation data : {len(self.val_images)}')
            self.logger.info(f'checking length of test data : {len(self.test_images)}')

            # Base directory
            self.base_dir = r"C:\INTERNSHIP_VIHARATECH\INTERN_PR2\Split_data"
     # Define dataset splits
            self.splits = {"Train": self.train_dir,"Validation": self.valid_dir,"Test": self.test_dir}
       # Process each split
            for self.split_name, self.split_dir in self.splits.items():
                self.logger.info(f"Processing: {self.split_name}")
    # Step 1: Remove old group folders before regrouping
                for self.folder in os.listdir(self.split_dir):
                    self.folder_path = os.path.join(self.split_dir, self.folder)
                    if os.path.isdir(self.folder_path) and self.folder.startswith("Group_"):
                        shutil.rmtree(self.folder_path)
                        self.logger.info(f"Removed old folder: {self.folder_path}")
            # Step 2: Get all class folders
                self.class_names = []
                for self.name in os.listdir(self.split_dir):
                    self.path = os.path.join(self.split_dir, self.name)
                    if os.path.isdir(self.path):
                        self.class_names.append(self.name)
                self.class_names.sort()
                self.total_classes = len(self.class_names)
                self.logger.info(f"Total classes found in {self.split_name}: {self.total_classes}")
              # Step 3: Create 11 groups (10×3 + 1×4)
                self.groups = []
                self.start = 0
                for self.i in range(10):
                    self.group_classes = self.class_names[self.start:self.start + 3]
                    if self.group_classes:
                        self.groups.append(self.group_classes)
                    self.start += 3

                self.remaining_classes = self.class_names[self.start:]
                if self.remaining_classes:
                    self.groups.append(self.remaining_classes)

                # Step 4: Move classes into group folders
                self.group_number = 1
                for self.group_classes in self.groups:
                    self.group_folder = os.path.join(self.split_dir, f"Group_{self.group_number}")
                    os.makedirs(self.group_folder, exist_ok=True)
                    for self.cls in self.group_classes:
                        self.src = os.path.join(self.split_dir, self.cls)
                        self.dst = os.path.join(self.group_folder, self.cls)
                        if os.path.exists(self.src):
                            shutil.move(self.src, self.dst)
                            self.logger.info(f"Moved {self.cls} to {self.group_folder}")
                    self.group_number += 1
                # Step 5: Verify grouping
                self.total_groups = len([
                    d for d in os.listdir(self.split_dir)
                    if os.path.isdir(os.path.join(self.split_dir, d)) and d.startswith("Group_")])
                self.logger.info(f"{self.split_name} data grouped into {self.total_groups} folders successfully.")
            self.logger.info("Grouping completed for Train, Validation, and Test with unique non-repeating classes.")
         # Step 6: Log the grouping structure
            for self.split_name, self.split_dir in self.splits.items():
                self.logger.info(f"------- {self.split_name} Data -------")

                self.group_folders = [d for d in os.listdir(self.split_dir)
                    if os.path.isdir(os.path.join(self.split_dir, d)) and d.startswith("Group_")]
                self.group_folders.sort(key=lambda x: int(x.split("_")[1]))
                for self.group_folder in self.group_folders:
                    self.group_path = os.path.join(self.split_dir, self.group_folder)
                    self.class_list = [c for c in os.listdir(self.group_path)
                        if os.path.isdir(os.path.join(self.group_path, c))]
                    self.class_list.sort()
                    self.class_count = len(self.class_list)
                    self.logger.info(f"{self.group_folder} -> Total Classes: {self.class_count}")
                    for self.cls in self.class_list:
                        self.logger.info(f"    - {self.cls}")

        except Exception as e:
            e_type, e_value, e_traceback = sys.exc_info()
            self.logger.error(f"Error in data_grouping on line {e_traceback.tb_lineno} because {e_value}")

    def final_validation(self):
        try:
            base_path = r"C:\INTERNSHIP_VIHARATECH\INTERN_PR2"
            image_path = r"C:\INTERNSHIP_VIHARATECH\INTERN_PR2\saved_images\033.jpg"
            model_folders = { "Custom Model": os.path.join(base_path, "custom_models"),
                                "ResNet-50": os.path.join(base_path, "resnet50_models"),
                                "VGG-16": os.path.join(base_path, "vgg16_models")}
            # --- Single image preprocessing ---
            img = image.load_img(image_path, target_size=(256, 256))
            img_array = image.img_to_array(img) / 255.0
            img_array_exp = np.expand_dims(img_array, axis=0)

            plt.figure(figsize=(5, 5))
            plt.imshow(img_array)
            plt.axis('off')
            plt.title("Input Image")
            plt.show()

            # --- Test image on all models ---
            pred_results = {}
            for model_name, folder in model_folders.items():
                model_file = sorted([f for f in os.listdir(folder) if f.endswith(".h5")])[0]
                model = load_model(os.path.join(folder, model_file))

                # Assuming Group_1 classes
                group_path = os.path.join(base_path, "Split_data", "Train", "Group_1")
                fixed_classes = sorted([d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))])
                pred = model.predict(img_array_exp, verbose=0)
                pred_class = fixed_classes[np.argmax(pred)]
                pred_results[model_name] = pred_class
                self.logger.info(f"{model_name} predicts: {pred_class}")
            # --- Optional: Overall accuracy comparison plot (dummy example, replace with real numbers) ---
            model_names = list(pred_results.keys())
            accuracies = [0.85, 0.88, 0.80]  # Example accuracies, replace with real if available

            plt.figure(figsize=(8, 5))
            plt.bar(model_names, accuracies, color=['skyblue', 'orange', 'green'])
            plt.ylim(0, 1)
            plt.ylabel("Accuracy")
            plt.title("Overall Accuracy Comparison")
            plt.show()

        except Exception as e:
            e_type, e_value, e_traceback = sys.exc_info()
            self.logger.error(f"Error in final_validation on line {e_traceback.tb_lineno} because {e_value}")

    def models_loading(self):
        try:
            self.logger.info("Starting evaluation of all .h5 models...")
            # Base path for models and datasets
            self.base_path = r"C:\INTERNSHIP_VIHARATECH\INTERN_PR2"
            # Folders containing the saved models
            self.model_folders = {"Custom Model": os.path.join(self.base_path, "custom_models"),
                                    "ResNet-50": os.path.join(self.base_path, "resNet50_models"),
                                    "VGG-16": os.path.join(self.base_path, "vgg16_models")}
            # Loop through each model type
            for self.model_name, self.folder in self.model_folders.items():
                self.logger.info("____________")
                self.logger.info(self.model_name)
                self.logger.info("____________" )
                self.model_files = sorted([f for f in os.listdir(self.folder) if f.endswith(".h5")])
                for self.idx, self.file_name in enumerate(self.model_files, start=1):
                    self.logger.info(f"\nGroup {self.idx}: {self.file_name}")
                    self.logger.info("____________")
                    self.model_path = os.path.join(self.folder, self.file_name)
                    self.model = load_model(self.model_path)
                   # Paths for each dataset
                    self.train_path = os.path.join(self.base_path, "Split_data", "Train", f"Group_{self.idx}")
                    self.val_path = os.path.join(self.base_path, "Split_data", "Validation", f"Group_{self.idx}")
                    self.test_path = os.path.join(self.base_path, "Split_data", "Test", f"Group_{self.idx}")
                   # Dynamically get classes from the train folder
                    self.fixed_classes = sorted([d for d in os.listdir(self.train_path)if os.path.isdir(os.path.join(self.train_path, d))])
                   # Image generators (dynamic classes)
                    self.train_gen = ImageDataGenerator(rescale=1 / 255).flow_from_directory(self.train_path, target_size=(256, 256),
                                        class_mode="categorical", shuffle=False,classes=self.fixed_classes)
                    self.val_gen = ImageDataGenerator(rescale=1 / 255).flow_from_directory(self.val_path, target_size=(256, 256),
                                     class_mode="categorical", shuffle=False,classes=self.fixed_classes)
                    self.test_gen = ImageDataGenerator(rescale=1 / 255).flow_from_directory(self.test_path,
                                    target_size=(256, 256),class_mode="categorical", shuffle=False,classes=self.fixed_classes)
                   # Predictions
                    self.train_pred = np.argmax(self.model.predict(self.train_gen), axis=1)
                    self.val_pred = np.argmax(self.model.predict(self.val_gen), axis=1)
                    self.test_pred = np.argmax(self.model.predict(self.test_gen), axis=1)

                    # True labels
                    self.y_train = self.train_gen.classes
                    self.y_val = self.val_gen.classes
                    self.y_test = self.test_gen.classes

                    # Accuracy
                    self.logger.info(f"Train Accuracy: {accuracy_score(self.y_train, self.train_pred):.4f}")
                    self.logger.info(f"Validation Accuracy: {accuracy_score(self.y_val, self.val_pred):.4f}")
                    self.logger.info(f"Test Accuracy: {accuracy_score(self.y_test, self.test_pred):.4f}")

                    # Confusion matrix
                    self.cm = confusion_matrix(self.y_test, self.test_pred)
                    self.logger.info("\nConfusion Matrix:")
                    self.logger.info("\n" + str(self.cm))

                    # Per-class metrics
                    self.logger.info("\nClass-wise Performance (TP / FP / FN / TN):")
                    for self.i, self.cls in enumerate(self.fixed_classes):
                        self.TP = self.cm[self.i, self.i]
                        self.FP = self.cm[:, self.i].sum() - self.TP
                        self.FN = self.cm[self.i, :].sum() - self.TP
                        self.TN = self.cm.sum() - (self.TP + self.FP + self.FN)
                        self.logger.info(f"{self.cls}  TP={self.TP}, FP={self.FP}, FN={self.FN}, TN={self.TN}")

                    # Classification report
                    self.logger.info("\nClassification Report:")
                    self.num_classes = len(self.fixed_classes)
                    self.logger.info("\n" + classification_report(self.y_test, self.test_pred,target_names=self.fixed_classes,
                                 labels=list(range(self.num_classes))))

            self.logger.info("All models evaluated successfully!")

        except Exception as e:
            e_type, e_value, e_traceback = sys.exc_info()
            self.logger.error(f"Error in models_loading on line {e_traceback.tb_lineno} because {e_value}")

    def performance_json(self):
        try:

            self.logger.info("===== STARTING PERFORMANCE JSON GENERATION =====")

            base_data_path = r"C:\INTERNSHIP_VIHARATECH\INTERN_PR2\Split_data"
            model_base_path = r"C:\INTERNSHIP_VIHARATECH\INTERN_PR2"
            save_json_path = r"C:\INTERNSHIP_VIHARATECH\INTERN_PR2\model_performance.json"
         # folders where your saved .h5 files live
            model_folders = { "VGG-16": os.path.join(model_base_path, "vgg16_models"),
                                "ResNet-50": os.path.join(model_base_path, "resnet50_models"),
                                "Custom Model": os.path.join(model_base_path, "custom_models")}
            # expected group names
            groups = [f"Group_{i}" for i in range(1, 12)]
            final_output = []
            import re
            def extract_group_num(filename):
                fname = filename.lower()
                # search patterns like group1 or group_1
                m = re.search(r'group[_\-]*0*([1-9][0-9]?)', fname)
                if m:
                    try:
                        return int(m.group(1))
                    except:
                        return None
                # fallback: search digits near 'group' word
                m2 = re.search(r'group.*?(\d{1,2})', fname)
                if m2:
                    try:
                        return int(m2.group(1))
                    except:
                        return None
                return None

            for model_name, model_path in model_folders.items():
                self.logger.info(f"Checking model folder: {model_name} -> {model_path}")
                if not os.path.exists(model_path):
                    self.logger.warning(f"Model folder missing: {model_path}. Skipping {model_name}.")
                    continue
                model_files_all = [f for f in os.listdir(model_path) if f.endswith(".h5")]
                if len(model_files_all) == 0:
                    self.logger.warning(f"No .h5 models found in {model_path}. Skipping {model_name}.")
                    continue
               # Build mapping: group_number -> filepath (pick first match if multiple)
                group_to_file = {}
                for f in model_files_all:
                    gnum = extract_group_num(f)
                    if gnum is not None and 1 <= gnum <= 11:
                        # if multiple files for same group, prefer exact pattern containing 'group' and index
                        if gnum not in group_to_file:
                            group_to_file[gnum] = os.path.join(model_path, f)
                # If mapping is incomplete, try matching files by substring group_x where possible
                # (still keep previously found ones)
                for idx, group in enumerate(groups, start=1):
                    self.logger.info(f"  Processing Group: {group}")
                    test_path = os.path.join(base_data_path, "Test", group)
                    if not os.path.exists(test_path):
                        self.logger.warning(f"  Test folder missing: {test_path}. Skipping group.")
                        continue
                 # find model file for this group
                    model_file = None
                    if idx in group_to_file:
                        model_file = group_to_file[idx]
                    else:
                        # fallback: try to find any file containing 'group' + idx or 'group_idx' ignoring case
                        target_patterns = [f"group{idx}", f"group_{idx}", f"group-{idx}", f"_{idx}."]
                        for f in model_files_all:
                            lf = f.lower()
                            if f"group{idx}" in lf or f"group_{idx}" in lf or f"group-{idx}" in lf or f"_group{idx}" in lf or f"_group_{idx}" in lf:
                                model_file = os.path.join(model_path, f)
                                break
                        # last fallback: if only one model and groups are in order, allow using file by index (not ideal)
                        # but we avoid that automatic fallback to prevent mismatched mapping.
                    if model_file is None:
                        self.logger.warning(f"  No model found for {group} in {model_path}.")
                        continue
                    self.logger.info(f"  Loading model: {model_file}")
                    try:
                        model = tf.keras.models.load_model(model_file)
                    except Exception as e:
                        self.logger.error(f"  Failed to load model {model_file}: {e}")
                        continue
                    # Determine target size to use for test generator.
                    # Prefer model.input_shape if valid (channels last), else default to (220,220)
                    default_size = (220, 220)
                    try:
                        inshape = model.input_shape  # e.g. (None, H, W, 3)
                        # handle cases where input_shape may be (None, 3, H, W) or (None, H, W, 3)
                        if inshape is None:
                            target_size = default_size
                        else:
                            if len(inshape) == 4:
                                # detect channels-last
                                if inshape[1] is not None and inshape[2] is not None:
                                    h = int(inshape[1])
                                    w = int(inshape[2])
                                    # sanity check: if dims look weird (like too large for 220), fallback to 220
                                    if 20 <= h <= 1024 and 20 <= w <= 1024:
                                        target_size = (h, w)
                                    else:
                                        target_size = default_size
                                # channels-first (None, 3, H, W)
                                elif inshape[2] is not None and inshape[3] is not None:
                                    h = int(inshape[2]);
                                    w = int(inshape[3])
                                    if 20 <= h <= 1024 and 20 <= w <= 1024:
                                        target_size = (h, w)
                                    else:
                                        target_size = default_size
                                else:
                                    target_size = default_size
                            else:
                                target_size = default_size
                    except Exception:
                        target_size = default_size

                    self.logger.info(f"  Using target image size: {target_size[0]}x{target_size[1]} for predictions")
                    try:
                        test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(test_path,
                                    target_size=target_size,batch_size=16,class_mode='categorical',shuffle=False)
                    except Exception as e:
                        self.logger.error(f"  Failed to create test generator for {test_path}: {e}")
                        continue
                    if getattr(test_gen, "samples", 0) == 0:
                        self.logger.warning(f"  No images found in test folder {test_path}. Skipping.")
                        continue
                    try:
                        preds = model.predict(test_gen, verbose=0)
                    except Exception as e:
                        self.logger.error(f"  Model prediction failed for {model_file} on {test_path}: {e}")
                        continue
                   # handle case of single-class (shouldn't happen) and multi-class
                    try:
                        y_pred = np.argmax(preds, axis=1)
                    except Exception:
                        # if preds are already labels
                        y_pred = preds.ravel().astype(int)
                    y_true = test_gen.classes
                    # confusion and report
                    try:
                        cm = confusion_matrix(y_true, y_pred)
                    except Exception as e:
                        self.logger.error(f"  Failed to compute confusion matrix for {group} with model {model_file}: {e}")
                        continue
                    # get class order from generator
                    # test_gen.class_indices maps class_name -> label index
                    label_to_class = {v: k for k, v in test_gen.class_indices.items()}
                    ordered_classes = [label_to_class[i] for i in sorted(label_to_class.keys())]
                    # classification_report
                    try:
                        report_dict = classification_report(y_true, y_pred, target_names=ordered_classes,output_dict=True)
                    except Exception as e:
                        self.logger.error(f"  Failed to produce classification report for {group}: {e}")
                        report_dict = {}
                    # compute test accuracy overall
                    try:
                        test_accuracy = float((y_true == y_pred).sum()) / len(y_true)
                    except:
                        test_accuracy = None
                    # Build group JSON
                    group_result = {
                        "Model_Name": model_name,
                        "Model_File": os.path.basename(model_file),
                        "Group_Name": group,
                        "Test_Path": test_path,
                        "Test_Samples": int(test_gen.samples),
                        "Test_Accuracy": round(test_accuracy * 100, 2) if test_accuracy is not None else None,
                        "Classes": [] }
                    # ensure cm is at least NxN for number of classes
                    num_classes = len(ordered_classes)
                    # If confusion_matrix returned smaller for some reason, pad
                    if cm.shape != (num_classes, num_classes):
                        # attempt to build full confusion matrix
                        full_cm = np.zeros((num_classes, num_classes), dtype=int)
                        try:
                            for t, p in zip(y_true, y_pred):
                                if 0 <= t < num_classes and 0 <= p < num_classes:
                                    full_cm[t, p] += 1
                            cm = full_cm
                        except:
                            # fallback: reshape if possible
                            cm = np.zeros((num_classes, num_classes), dtype=int)
                    for i, cname in enumerate(ordered_classes):
                        tp = int(cm[i, i])
                        fn = int(cm[i, :].sum() - tp)
                        fp = int(cm[:, i].sum() - tp)
                        tn = int(cm.sum() - (tp + fp + fn))
                     # metrics from classification_report if available
                        precision = None;
                        recall = None;
                        f1 = None;
                        support = None
                        if cname in report_dict:
                            try:
                                precision = round(report_dict[cname].get("precision", 0.0) * 100, 2)
                                recall = round(report_dict[cname].get("recall", 0.0) * 100, 2)
                                f1 = round(report_dict[cname].get("f1-score", 0.0) * 100, 2)
                                support = int(report_dict[cname].get("support", 0))
                            except:
                                precision = recall = f1 = None
                        class_entry = {
                            "Class_Name": cname,
                            "Support": support if support is not None else int(test_gen.class_indices.get(cname, 0)),
                            "Precision(%)": precision,
                            "Recall(%)": recall,
                            "F1(%)": f1,
                            "TP": tp,
                            "FP": fp,
                            "FN": fn,
                            "TN": tn}
                        group_result["Classes"].append(class_entry)
                    final_output.append(group_result)
                    self.logger.info(f"  Completed evaluation for {model_name} - {group}")
            # Save JSON
            try:
                with open(save_json_path, "w", encoding="utf-8") as wf:
                    json.dump(final_output, wf, indent=4, ensure_ascii=False)
                self.logger.info(f"===== JSON CREATED: {save_json_path} =====")
            except Exception as e:
                self.logger.error(f"Failed to save JSON to {save_json_path}: {e}")

        except Exception as e:
            e_type, e_value, e_traceback = sys.exc_info()
            self.logger.error(f"Error in performance_json on line {e_traceback.tb_lineno} because {e_value}")


if __name__ == '__main__':
    try:
        path = r'C:\INTERNSHIP_VIHARATECH\INTERN_PR2\Food Classification dataset'
        obj = FOOD_CLASSIFICATION(path)
        obj.check_classnames()
        obj.json_data()
        obj.selected_images()
        obj.splitting_data()
        obj.data_grouping()
        obj.final_validation()
        obj.models_loading()
        obj.performance_json()


    except Exception as e:
        e_type, e_value, e_traceback = sys.exc_info()
        logger.error(f"Error in __main__ on line {e_traceback.tb_lineno} because {e_value}")
