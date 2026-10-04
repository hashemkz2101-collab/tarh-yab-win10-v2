import os, sys
from transformers import AutoImageProcessor, AutoModel

MODELS = (("facebook/dinov2-small", "model_dinov2_small"),
          ("facebook/dinov2-base", "model_dinov2_base"))

for name, folder in MODELS:
    if os.path.exists(os.path.join(folder, "model.safetensors")):
        print("already downloaded:", folder)
        continue
    try:
        AutoImageProcessor.from_pretrained(name).save_pretrained(folder)
        AutoModel.from_pretrained(name).save_pretrained(folder)
        print("OK:", folder)
    except Exception as e:
        print("SKIP:", name, e)

# the small model is required; the base model is optional
sys.exit(0 if os.path.exists(os.path.join("model_dinov2_small", "model.safetensors")) else 1)
