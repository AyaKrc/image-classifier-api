import torch
import timm
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load labels
with open("artifacts/labels.json", "r") as f:
    CLASS_NAMES = json.load(f)

def load_model():
    model = timm.create_model("convnextv2_base.fcmae_ft_in1k", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load("artifacts/best_model.pth", map_location=device))
    model.eval().to(device)
    return model, CLASS_NAMES, device
