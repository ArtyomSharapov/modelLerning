from ultralytics import YOLO
import cv2
model = YOLO(
    "models/segmentationModel.pt"
)

#проверка работоспособности сегментации
results = model.predict(
    source="imagesForTest/images/name.jpg",
    imgsz=320,
    save=False,
    exist_ok=True,
    workers=0)

result = results[0]
image = result.plot()
image_save = "imagesForTest/resultImages/name.jpg"
cv2.imwrite(image_save,image)
print("Маска сохранена в:", image_save)
