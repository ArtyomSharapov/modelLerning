from ultralytics import YOLO

model = YOLO(
    r"D:\projects\python\modelLerning\runs\semantic\crack_seg_GPU-5\weights\best.pt"
)

model.predict(
    source=r"D:\projects\python\modelLerning\datasets\dataset_segmentation_yolo\images\val\image_783.jpg",
    imgsz=320,
    save=True,
    project=r"D:\projects\python\modelLerning\other/",
    name = "sss",
    workers=0)