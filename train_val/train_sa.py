# !pip install -e ultralytics-d8701b42caeb9f7f1de5fd45e7c3f3cf1724ebb6/.

from ultralytics import YOLO

model = YOLO(model="yolov8-seg.yaml")
model.train(data="sa.yaml", \
            epochs=50, \
            batch=32, \
            imgsz=1024, \
            overlap_mask=False, \
            save=True, \
            save_period=5, \
            device='0',\
            workers=16,\
            project='fastsam', \
            name='Training with 15 classes', \
            val=True, \
            lr0=0.01, \
            lrf=0.01, \
            seed=0, \
            pretrained=False,)
