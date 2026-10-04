# -*- coding: utf-8 -*-
"""بارگذاری مدل و ساخت بردار تصویر (بدون streamlit؛ برای evaluate.py)"""
import os
import numpy as np
from PIL import Image

MODELS = {
    "small": dict(local="model_dinov2_small", hub="facebook/dinov2-small"),
    "base":  dict(local="model_dinov2_base",  hub="facebook/dinov2-base"),
}


def make_embedder(key="small"):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    m = MODELS[key]; src = m["local"] if os.path.isdir(m["local"]) else m["hub"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(src)
    model = AutoModel.from_pretrained(src).to(device).eval()

    def embed(tiles, progress=None):
        out, n = [], len(tiles)
        with torch.inference_mode():
            for i in range(0, n, 32):
                batch = [Image.fromarray(t).convert("RGB") for t in tiles[i:i + 32]]
                inp = proc(images=batch, size={"shortest_edge": 224}, do_center_crop=False,
                           return_tensors="pt").to(device)
                h = model(**inp).last_hidden_state
                v = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
                out.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
                if progress: progress(min(i + 32, n), n)
        return np.vstack(out).astype("float32")
    return embed
