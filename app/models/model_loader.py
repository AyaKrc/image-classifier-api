import os
import torch
import timm
import json

# CRITICAL: Set these BEFORE importing huggingface_hub
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = "/tmp/huggingface"

# Now import huggingface_hub
from huggingface_hub import hf_hub_download

def load_model():
    """Load the ConvNextV2 model with pretrained weights"""
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Load class names
    labels_path = "artifacts/labels.json"
    try:
        with open(labels_path, "r") as f:
            CLASS_NAMES = json.load(f)
        print(f"✅ Loaded {len(CLASS_NAMES)} class labels")
    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        CLASS_NAMES = ["REAL", "FAKE"]  # Fallback labels
    
    # Create cache directory with proper permissions
    cache_dir = "/tmp/huggingface_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        print("📥 Downloading model from Hugging Face Hub...")
        
        # Download model with specific cache directory
        model_path = hf_hub_download(
            repo_id="AyaKrc/image-classifier-cifake",
            filename="best_model.pth",
            cache_dir=cache_dir,
            local_dir_use_symlinks=False,  # Avoid symlink issues
            resume_download=True  # Resume if interrupted
        )
        
        print(f"✅ Model downloaded to: {model_path}")
        
    except Exception as e:
        print(f"❌ Download error: {e}")
        
        # Alternative: Try direct download without Xet
        import urllib.request
        model_url = "https://huggingface.co/AyaKrc/image-classifier-cifake/resolve/main/best_model.pth"
        model_path = os.path.join(cache_dir, "best_model.pth")
        
        try:
            print("🔄 Attempting direct download...")
            urllib.request.urlretrieve(model_url, model_path)
            print(f"✅ Direct download successful: {model_path}")
        except Exception as e2:
            print(f"❌ Direct download failed: {e2}")
            raise RuntimeError("Failed to download model from Hugging Face")
    
    # Create model architecture
    print("🏗️ Creating model architecture...")
    model = timm.create_model(
        "convnextv2_base.fcmae_ft_in1k",
        pretrained=False,  # We'll load our own weights
        num_classes=2
    )
    
    # Load weights
    try:
        print("📦 Loading model weights...")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        raise
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"🚀 Model ready on {device}")
    return model, CLASS_NAMES, device