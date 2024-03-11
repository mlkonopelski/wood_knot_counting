from ultralytics import YOLO

# Load a model
model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)

model.train(data="dataset.yaml",
            cfg='training.yaml',
            name='yolov8l_1920_transaug')