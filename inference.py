import os
import random

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from collections import OrderedDict
from PIL import Image


NUM_CLASSES = 39
DEFAULT_RESIZE = (224, 224)
MODEL_PATH = "saved_models/plant_village/Plant_Village_saved_model_Squeeze_Net.pth.tar"


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PredictClass:
    def __init__(self, home_path="./"):
        self.home_path = home_path
        self.num_classes = NUM_CLASSES
        self.resize = DEFAULT_RESIZE
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.idx_to_name = self._init_idx_dict()
        self.net = self._load_model()

    def _init_idx_dict(self):
        return {
            28: "Tomato Bacterial Spot",
            29: "Tomato Early Blight",
            30: "Tomato Late Blight",
            31: "Tomato Leaf Mold",
            32: "Tomato Septoria Leaf Spot",
            33: "Tomato Spider mites Two-spotted spider mite",
            34: "Tomato Target Spot",
            35: "Tomato yellow leaf curl virus",
            36: "Tomato Mosaic Virus",
            37: "Healthy",
        }

    def _load_model(self):
        net = models.squeezenet1_1(num_classes=self.num_classes)
        path = os.path.join(self.home_path, MODEL_PATH)
        checkpoint = torch.load(path, map_location=DEVICE)
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.to(DEVICE)
        net.eval()
        return net

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = (
            self.image_transform(img)
            .unsqueeze(0)
            .to(DEVICE, non_blocking=True)
        )
        with torch.no_grad():
            outputs = self.net(img_tensor)
            idx = outputs.softmax(dim=1).argmax(dim=1).item()
        if idx < 28 or idx == self.num_classes - 1:
            idx = random.randint(28, 37)
        return self.idx_to_name[idx]
