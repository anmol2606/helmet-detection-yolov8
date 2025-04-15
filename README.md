# Helmet Detection using YOLOv8 

This project is a **Helmet Detection System** using the YOLOv8 object detection model. It detects whether a person is wearing a helmet using custom-trained YOLOv8 weights. The dataset used is from Kaggle and has been preprocessed and converted into the YOLO format for training.

# Project Structure

helmet-detection-yolov8/
|----dataset.yaml
|----annotations
|----images
|      |----BikesHelmets0.png
|     └── ...
├──dataset 
|      |----images/
│     |      ├── train/
│     |      │       ├── img1.jpg
│     |      │       └── ...
│     |      └── val/
│     |              ├── img2.jpg
│     |              └── ...
|      |---- labels/
│             ├── train/
│             │    ├── img1.txt
│             │    └── ...
│             └── val/
│                    ├── img2.txt
│                    └── ...
|----vol_to_yolo.py
|----helmet_detected.py

---

# Dataset

- **Source:** [Helmet Detection Dataset - Kaggle](https://www.kaggle.com/datasets/andrewmvd/helmet-detection)
- **Format:** The dataset is in **Pascal VOC** format with `.xml` annotations.
- **Content:** The dataset contains images and annotations of people with and without helmets.

# Steps to Get the Dataset:

1. Go to [Kaggle - Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/helmet-detection).
2. Download the dataset as a ZIP file and extract it.

---

# Convert VOC to YOLO Format

Before training the model, we need to convert the dataset annotations from **Pascal VOC** format to the **YOLO format**.

# Run the Conversion Script:

1. Download the dataset as `.xml` annotation files (Pascal VOC format).
2. Run the conversion script `voc_to_yolo.py` to convert the annotations to YOLO format.

# voc_to_yolo.py does the following:

-Converts Pascal VOC .xml files into YOLO .txt format.
-Organizes the images and labels into proper directories.
-The converted labels will be stored in a labels/ folder.

After conversion, your folder structure should look like this:

Helmet-Detection-Dataset/
├── Annotations/    # Contains VOC .xml annotations (original)
├── JPEGImages/     # Contains image files (JPEG format)
├── labels/         # Contains YOLO format .txt label files
└── dataset.yaml    # YOLO configuration file

---

# training the YOLOv8 Model
Now that we have our dataset in YOLO format, we can train the YOLOv8 model on it.

# Steps to Train YOLOv8:
-Install the necessary dependencies: Ensure you have ultralytics installed.
-Create the dataset.yaml file for YOLOv8, pointing to the correct paths for images and labels:

Example dataset.yaml:

path: ../helmet-dataset  # Path to the folder containing the images and labels
train: images/train      # Path to training images
val: images/val          # Path to validation images
nc: 2                    # Number of classes (Helmet, No Helmet)
names: ['Helmet', 'No Helmet']  # List of class names

# Train the model using YOLOv8:

-Run the following command to start training:
-yolo detect train model=yolov8s.pt data=dataset.yaml epochs=50 imgsz=640 batch=16

model=yolov8s.pt: The YOLOv8 small model is used as the base model.
data=dataset.yaml: Points to the dataset configuration file.
epochs=50: Number of training epochs.
imgsz=640: Image size used during training.
batch=16: Number of images in each batch.

Trained model weights (best.pt) will be saved in the runs/detect/train/weights/ directory.

---

# Real-time Inference with the Trained Model

-Once the model is trained, you can perform real-time inference using the script helmet_detection.py.

-Steps to Run Real-time Detection:
Ensure you have the trained best.pt weights in the same directory as helmet_detection.py.
Run the inference script for real-time helmet detection on webcam or video stream.

Output: The script will display real-time predictions on the webcam feed or video, showing whether a person is wearing a helmet or not.
