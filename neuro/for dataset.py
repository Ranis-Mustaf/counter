from ultralytics import YOLO

# Загрузка модели YOLOv8n
model = YOLO('yolov8n.pt')

# Обучение модели на твоих данных
model.train(data='C:/Users/mustafin_rf/Desktop/neuro/dataset/data.yaml', epochs=20, imgsz=640)
results = model("C:/Users/mustafin_rf/Desktop/neuro/dataset/test/images/1.jpg")
model.export(format='onnx')
