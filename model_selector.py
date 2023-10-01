import torch
import os
from ultralytics import YOLO

class model_generator:

    def __init__(self, model_type, model_name):
        self.model_type = model_type
        self.model_name = model_name
        self.dir = os.getcwd()

    def get_model(self):
        if self.model_type == 'ResNet':
            model = torch.hub.load(
                'pytorch/vision:v0.10.0',
                str(self.model_name),
                pretrained=True,
                progress=True
            )
        
        if self.model_type == 'EfficientNet':
            model = torch.hub.load(
                'pytorch/vision',
                self.model_name,
                pretrained=True
            )

        if self.model_type == 'MobileNet':
            model = torch.hub.load(
                'pytorch/vision:v0.10.0',
                self.model_name,
                pretrained=True
            )

        if self.model_type == 'YOLOv8':
            model = YOLO(self.dir + "/weights/" + self.model_name)

        return model


