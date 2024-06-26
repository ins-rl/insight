from ultralytics import YOLO
model = YOLO(model="yolov8x-seg.yaml", \
             )
model.train(data="sa.yaml", \
            epochs=100, \
            batch=32, \
            imgsz=1024, \
            overlap_mask=False, \
            save=True, \
            save_period=5, \
            device='0',\
            project='fastsam', \
            name='test', 
            val=False,)