from ultralytics import YOLO

#модель для классификации(есть ли трещина)
cls_model = YOLO("yolov8n-cls.pt") 
cls_model.train(
    data="datasets/Dataset_classification_ready",
    epochs=10,
    imgsz=320,
    batch=16,
    device=0,
    workers=0,
    name="crack_or_notCrack")


