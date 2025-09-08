import torch
import timm
import json
import os
from huggingface_hub import hf_hub_download

# Force disable Xet (avoids permission issues on Spaces)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load labels
with open("artifacts/labels.json", "r") as f:
    CLASS_NAMES = json.load(f)

def load_model():
    # Always use a writable cache dir
    cache_dir = "/tmp/huggingface"

    model_path = hf_hub_download(
        repo_id="AyaKrc/image-classifier-cifake",
        filename="best_model.pth",
        cache_dir=cache_dir,
        force_download=True  # ✅ ensures it retries fresh
    )

    # Log where the model was downloaded
    print(f"✅ Model downloaded and cached at: {model_path}")

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

    print("✅ Model loaded and ready for inference")
    return model, CLASS_NAMES, device
