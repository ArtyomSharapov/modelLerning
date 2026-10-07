from ultralytics import YOLO

#модель для классификации(есть ли трещина)
cls_model = YOLO("yolov8n-cls.pt") 
cls_model.train(
    data="path_for_your_dataset",
    epochs=10,
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    name="name_your_model")


