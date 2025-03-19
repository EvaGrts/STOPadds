import pandas as pd
import os

# Charger les annotations Open Images
annotations_train = pd.read_csv("C:/Users/peron/fiftyone/open-images-v7/train/labels/detections.csv")
annotations_val = pd.read_csv("C:/Users/peron/fiftyone/open-images-v7/validation/labels/detections.csv")
annotations_test = pd.read_csv("C:/Users/peron/fiftyone/open-images-v7/test/labels/detections.csv")
# Filtrer uniquement les Billboards
billboard_id = "/m/01knjb"  # ID de la classe Billboard dans Open Images


def convert2yolo(annotations,split):
    annotations = annotations[annotations["LabelName"] == billboard_id]
    # Convertir en format YOLO
    for _, row in annotations.iterrows():
        image_name = row["ImageID"] + ".jpg"
        txt_filename = os.path.join("C:/Users/peron/fiftyone/open-images-v7-YOLOformat/labels/"+split+"/"+ row["ImageID"] + ".txt")
        
        # Normalisation des coordonnées bbox
        x_center = (row["XMin"] + row["XMax"]) / 2
        y_center = (row["YMin"] + row["YMax"]) / 2
        width = row["XMax"] - row["XMin"]
        height = row["YMax"] - row["YMin"]

        with open(txt_filename, "w") as f:
            f.write(f"0 {x_center} {y_center} {width} {height}\n")  

convert2yolo(annotations_train,"train")
convert2yolo(annotations_val,"validation")
convert2yolo(annotations_test,"test")