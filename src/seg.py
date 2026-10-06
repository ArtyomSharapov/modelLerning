from ultralytics import YOLO


seg_model = YOLO("yolo26n-sem.pt")
seg_model.train(
    data="datasets/dataset_segmentation_yolo/data.yaml",
    epochs=10,
    imgsz=512,
    batch=4,
    patience=5,
    device=0,
    workers=0,
    name="crack_seg_GPU")
    
