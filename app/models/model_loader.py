import torch
import timm
import json
from huggingface_hub import hf_hub_download

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load labels (these stay local, small file)
with open("artifacts/labels.json", "r") as f:
    CLASS_NAMES = json.load(f)

def load_model():
    # Download model weights from Hugging Face Hub
    model_path = hf_hub_download(
        repo_id="AyaKrc/image-classifier-cifake",  # your HF repo
        filename="best_model.pth"
    )

    # Recreate architecture
    model = timm.create_model(
        "convnextv2_base.fcmae_ft_in1k",
        pretrained=False,
        num_classes=2
    )

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    # Prepare for inference
    model.eval().to(device)

    return model, CLASS_NAMES, device
