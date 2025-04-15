import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import shutil

# ==== FIXED BASE DIR ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Points to 'helmet-detection-yolov8'
IMG_DIR = os.path.join(BASE_DIR, 'images')
ANN_DIR = os.path.join(BASE_DIR, 'annotations')

# ==== OUTPUT FOLDERS ====
OUT_IMG_TRAIN = os.path.join(BASE_DIR, 'dataset/images/train')
OUT_IMG_VAL = os.path.join(BASE_DIR, 'dataset/images/val')
OUT_LABEL_TRAIN = os.path.join(BASE_DIR, 'dataset/labels/train')
OUT_LABEL_VAL = os.path.join(BASE_DIR, 'dataset/labels/val')

CLASSES = ['with helmet', 'without helmet']  # <-- Your actual class names from XMLs

# ==== CREATE FOLDERS ====
for folder in [OUT_IMG_TRAIN, OUT_IMG_VAL, OUT_LABEL_TRAIN, OUT_LABEL_VAL]:
    os.makedirs(folder, exist_ok=True)

# ==== SPLIT DATA ====
image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

# ==== BBOX CONVERSION ====
def convert_bbox(size, box):
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x * dw, y * dh, w * dw, h * dh

# ==== CONVERT SINGLE ANNOTATION ====
def convert_annotation(img_file, mode):
    name, _ = os.path.splitext(img_file)
    xml_path = os.path.join(ANN_DIR, f'{name}.xml')
    if not os.path.exists(xml_path): return

    out_label_path = os.path.join(
        OUT_LABEL_TRAIN if mode == 'train' else OUT_LABEL_VAL, f'{name}.txt'
    )

    with open(out_label_path, 'w') as out_file:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        if size is None: return

        w, h = int(size.find('width').text), int(size.find('height').text)

        for obj in root.findall('object'):
            cls = obj.find('name').text.strip().lower()
            cls = cls.replace("without helmet", "without helmet").replace("with helmet", "with helmet")

            if cls not in CLASSES:
                continue  # silently skip unknown class

            cls_id = CLASSES.index(cls)
            xml_box = obj.find('bndbox')
            box = (
                int(xml_box.find('xmin').text),
                int(xml_box.find('ymin').text),
                int(xml_box.find('xmax').text),
                int(xml_box.find('ymax').text),
            )
            bb = convert_bbox((w, h), box)
            out_file.write(f"{cls_id} " + " ".join(map(str, bb)) + "\n")

# ==== PROCESS ALL FILES ====
for mode, files in [('train', train_files), ('val', val_files)]:
    for file in files:
        # Copy image
        shutil.copy(
            os.path.join(IMG_DIR, file),
            os.path.join(OUT_IMG_TRAIN if mode == 'train' else OUT_IMG_VAL, file)
        )
        # Convert annotation
        convert_annotation(file, mode)

print("✅ VOC to YOLO conversion completed successfully!")
