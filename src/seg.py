from ultralytics import YOLO

#обучение для сегментации
seg_model = YOLO("yolo26n-sem.pt")
seg_model.train(
    data="path_for_your_dataset_data.yaml",
    epochs=10,
    imgsz=512,
    batch=4,
    patience=5,
    device=0,
    workers=0,
    name="your_name")
    
