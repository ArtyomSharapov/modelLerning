from ultralytics import YOLO

model = YOLO(
    r"D:\projects\python\modelLerning\runs\semantic\crack_seg_GPU-2\weights\best.pt"
)

metrics = model.val(
    data=r"D:\projects\python\modelLerning\datasets\dataset_segmentation_yolo\data.yaml",
    workers=0
)


print("mIoU:", metrics.miou)
print("Pixel Accuracy:", metrics.pixel_accuracy)
print("IoU по классам:", metrics.per_class_iou)
print("Pixel Accuracy по классам:", metrics.per_class_pixel_accuracy)