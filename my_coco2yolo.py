import json
import os
import shutil
from tqdm import tqdm

coco_path = "F:/yolo/ultralytics-main/ultralytics-main/instance_version"
output_path = "/instance_version"

os.makedirs(os.path.join(output_path, "images", "train"), exist_ok=True)
os.makedirs(os.path.join(output_path, "images", "val"), exist_ok=True)
os.makedirs(os.path.join(output_path, "labels", "train"), exist_ok=True)
os.makedirs(os.path.join(output_path, "labels", "val"), exist_ok=True)

with open(os.path.join(coco_path, "train", "instances_train_trashcan.json"), "r") as f:
    train_annotations = json.load(f)

with open(os.path.join(coco_path, "val", "instances_val_trashcan.json"), "r") as f:
    val_annotations = json.load(f)

# Iterate over the training images
for image in tqdm(train_annotations["images"]):
    width, height = image["width"], image["height"]
    scale_x = 1.0 / width
    scale_y = 1.0 / height

    label = ""
    for annotation in train_annotations["annotations"]:
        if annotation["image_id"] == image["id"]:
            # Convert the annotation to YOLO format
            x, y, w, h = annotation["bbox"]
            x_center = x + w / 2.0
            y_center = y + h / 2.0
            x_center *= scale_x
            y_center *= scale_y
            w *= scale_x
            h *= scale_y
            class_id = annotation["category_id"]
            label += "{} {} {} {} {}\n".format(class_id, x_center, y_center, w, h)

    # Save the image and label
    shutil.copy(os.path.join(coco_path, "train", image["file_name"]),
                os.path.join(output_path, "images", "train", image["file_name"]))
    with open(os.path.join(output_path, "labels", "train", image["file_name"].replace(".jpg", ".txt")), "w") as f:
        f.write(label)

# Iterate over the validation images
for image in tqdm(val_annotations["images"]):
    width, height = image["width"], image["height"]
    scale_x = 1.0 / width
    scale_y = 1.0 / height

    label = ""
    for annotation in val_annotations["annotations"]:
        if annotation["image_id"] == image["id"]:
            # Convert the annotation to YOLO format
            x, y, w, h = annotation["bbox"]
            x_center = x + w / 2.0
            y_center = y + h / 2.0
            x_center *= scale_x
            y_center *= scale_y
            w *= scale_x
            h *= scale_y
            class_id = annotation["category_id"]
            label += "{} {} {} {} {}\n".format(class_id, x_center, y_center, w, h)

    # Save the image and label
    shutil.copy(os.path.join(coco_path, "val", image["file_name"]),
                os.path.join(output_path, "images", "val", image["file_name"]))
    with open(os.path.join(output_path, "labels", "val", image["file_name"].replace(".jpg", ".txt")), "w") as f:
        f.write(label)
