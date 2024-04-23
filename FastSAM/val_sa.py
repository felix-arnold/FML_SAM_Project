from ultralytics import YOLO
model = YOLO(model="./fastsam/test55/weights/best.pt")
model.val(data="sa.yaml", \
            epochs=20, \
            batch=1, \
            imgsz=1024, \
            device='0',\
            project='fastsam', \
            name='val', 
            val=False,
            save_json=True, \
            conf=0.001, \
            iou=0.9, \
            max_det=100, \
            )

from ultralytics import YOLO
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("cat_dog.png")

model = YOLO('./fastsam/test75/weights/best.pt')
model.info()

yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

results = model.predict(img)
colors = [random.choices(range(256), k=3) for _ in classes_ids]

overlay = np.zeros_like(img)
for result in results:
    for mask, box in zip(result.masks.xy, result.boxes):
        points = np.int32([mask])
        color_number = classes_ids.index(int(box.cls[0]))
        cv2.fillPoly(overlay, points, colors[color_number])

result = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

plt.imshow(result)
plt.show()
result.save("./output/cat_dog_bis.png)

