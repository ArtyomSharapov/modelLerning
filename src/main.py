from ultralytics import YOLO
import cv2
import numpy as np

cls_model = YOLO("models/classificationModel.pt")
seg_model = YOLO("models/segmentationModel.pt")

#путь для нашего изображения
image = "imagesForTest/images/name.jpg"

# процесс классификации
cls_result = cls_model(image, verbose=False)[0]

#получаем инфо о наличии трещины 
class_id = cls_result.probs.top1
class_name = cls_result.names[class_id]
confidence = float(cls_result.probs.top1conf)
print(f"Class: {class_name}")
print(f"Confidence: {confidence:.2f}")


if class_name == "Cracked":

    seg_result = seg_model(image,imgsz=512,verbose=False)[0]

    # маска
    maska = seg_result.semantic_mask.data
    maska = maska.cpu().numpy().squeeze()

    crack_maska = (maska == 1).astype(np.uint8)

    #наша фотка
    image = cv2.imread(image)

    #перобразуем размеры маски
    crack_maska = cv2.resize(crack_maska,(image.shape[1], image.shape[0]),interpolation=cv2.INTER_NEAREST)

    #оутпут -  выделенная трещина
    overlay = image.copy()
    overlay[crack_maska == 1] = (0, 0, 255)
    result = cv2.addWeighted(image,0.7,overlay,0.3,0)
    
    
    #сохранение результата, можем указать имя какое хотим 
    image_save = "imagesForTest/resultImages/name.jpg"
    cv2.imwrite(image_save, result)

    print("Трещина найдена и выделена")
    print(f"Маска была сохранена: {image_save}")

else:
    print("Трещины нет")