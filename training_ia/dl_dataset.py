import fiftyone as fo
import fiftyone.zoo as foz

# Télécharger uniquement les images contenant des Billboards
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split='train',
    label_types=["detections"],  # Chargement des bounding boxes
    classes=["Billboard"],       # Filtrer uniquement les images avec Billboards
)

dataset.name = "billboard_train"
dataset.persistent = True

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split='validation',
    label_types=["detections"], 
    classes=["Billboard"],      
)

dataset.name = "billboard_val"
dataset.persistent = True

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split='test',
    label_types=["detections"],  
    classes=["Billboard"],       
)


dataset.name = "billboard_test"
dataset.persistent = True
