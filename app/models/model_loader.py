import torch
import timm
import json
from huggingface_hub import hf_hub_download
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load labels
with open("artifacts/labels.json", "r") as f:
    CLASS_NAMES = json.load(f)

def load_model():
    # ✅ Ensure writable cache dir
    cache_dir = "/tmp/huggingface"
    os.makedirs(cache_dir, exist_ok=True)

    # ✅ Disable Xet storage (forces normal download)
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    # Download model weights
    model_path = hf_hub_download(
        repo_id="AyaKrc/image-classifier-cifake",  # your repo
        filename="best_model.pth",
        cache_dir=cache_dir
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

    model.eval().to(device)
    return model, CLASS_NAMES, device
