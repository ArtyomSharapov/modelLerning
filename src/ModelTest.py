from ultralytics import YOLO

best_model = YOLO("runs/classify/crack_or_notCrack-4/weights/best.pt")

#создаем переменную для одного фото и передаем в result
results = best_model("image_C_936.jpg")
result = results[0]

# тут получаем самый вероятный класс. 0/1
class_id = result.probs.top1
class_name = result.names[class_id]
# а тут считаем саму вероятность 
confidence = result.probs.top1conf.item()


print("Class:", class_name)
print("Confidence:", confidence)