import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class YOLOv8:
    def __init__(self,
                 model_def='yolov8n.pt',
                 model_folder='./models_/detectors/yolov8',
                 image_resolution=(640, 640),
                 conf_thres=0.3,
                 device=torch.device('cpu')):

        self.model_def = model_def
        self.model_folder = model_folder
        self.image_resolution = image_resolution
        self.conf_thres = conf_thres
        self.device = device

        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        # Load YOLOv8 model
        print(f"Loading YOLOv8 model: {self.model_def}")
        self.model = YOLO(self.model_def)
        self.model.fuse()
        self.model.to(self.device)

    def predict_single(self, image, color_mode='BGR'):
        # Convert BGR → RGB if necessary
        if color_mode == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run YOLOv8 detection
        results = self.model.predict(image, verbose=False, device=0 if self.device.type == 'cuda' else 'cpu')

        # Collect detections (xyxy, conf, cls)
        detections = []
        for box in results[0].boxes:
            conf = float(box.conf)
            cls = int(box.cls)
            if conf >= self.conf_thres and cls == 0:  # class 0 = person
                xyxy = box.xyxy.cpu().numpy().flatten()
                detections.append([*xyxy, conf, conf, cls])  # mimic YOLOv5 format w/ fake conf2

        if len(detections) == 0:
            return torch.empty((0, 7))

        return torch.tensor(detections, device=self.device)

    def predict(self, images, color_mode='BGR'):
        raise NotImplementedError("Batch predict not implemented.")
