from ultralytics import YOLO

if __name__ == '__main__':
    # Charger YOLOv8 avec un modèle pré-entraîné
    model = YOLO("yolov8s-oiv7.pt")  

    # Entraînement su GPU
    model.train(data="training_ia/data.yaml", epochs=50, batch=16, imgsz=640, device="cuda")


    metrics = model.val()
 
    #Export format TensorRT pour gpu
    model.export(format="engine")
    #Export format ONNX pour cpu
    model.export(format="onnx")
