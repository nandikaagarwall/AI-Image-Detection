"""
Pretrained open-source AI-image detectors (PyTorch), behind one interface:

    det = load_detector("commfor")          # downloads weights once into the Hugging Face cache
    p_ai = det.predict([pil_img1, pil_img2])  # -> np.array of P(AI-generated)

Weights are cached in %USERPROFILE%\\.cache\\huggingface\\hub (delete a folder there to free space).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

# key -> (Hugging Face repo, labels that mean "AI-generated", short description)
REGISTRY: dict[str, dict] = {
    "commfor": dict(
        repo="OwensLab/commfor-model-384", ai_labels=None, kind="commfor",
        desc="Community Forensics ViT-S/16-384 (Park & Owens, CVPR 2025; trained on 4,803 generators)",
    ),
    "umm-maybe": dict(
        repo="umm-maybe/AI-image-detector", ai_labels=["artificial"], kind="hf",
        desc="Swin, trained mostly on AI art (2022)",
    ),
    "organika-sdxl": dict(
        repo="Organika/sdxl-detector", ai_labels=["artificial"], kind="hf",
        desc="Swin, umm-maybe fine-tuned on SDXL vs Wikimedia photos",
    ),
    "haywoodsloan": dict(
        repo="haywoodsloan/ai-image-detector-deploy", ai_labels=["artificial"], kind="hf",
        desc="SwinV2, continuously retrained (2024-2026)",
    ),
    "ateeqq-siglip": dict(
        repo="Ateeqq/ai-vs-human-image-detector", ai_labels=["ai"], kind="hf",
        desc="SigLIP fine-tuned on 60k AI + 60k human images",
    ),
    "smogy": dict(
        repo="Smogy/SMOGY-Ai-images-detector", ai_labels=["artificial"], kind="hf",
        desc="Swin fine-tuned on 50k+ images",
    ),
    "nyuad": dict(
        repo="NYUAD-ComNets/NYUAD_AI-generated_images_detector", ai_labels=["dalle", "sd"], kind="hf",
        desc="ViT, 3 classes (dalle / sd / real)",
        # its preprocessor_config.json uses the legacy "ViTFeatureExtractor" name (removed in
        # transformers v5); these are the exact values from that file
        vit_processor=dict(size={"height": 224, "width": 224}, image_mean=[0.5] * 3,
                           image_std=[0.5] * 3, resample=2),
    ),
    "dima806-cifake": dict(
        repo="dima806/ai_vs_real_image_detection", ai_labels=["FAKE"], kind="hf",
        desc="ViT fine-tuned on CIFAKE (same dataset as your models)",
    ),
}


@dataclass
class Detector:
    key: str
    repo: str
    desc: str
    model: object
    processor: object = None
    ai_idx: list[int] = field(default_factory=list)
    kind: str = "hf"
    load_seconds: float = 0.0

    def predict(self, images: list[Image.Image], batch_size: int = 8) -> np.ndarray:
        import torch

        out = []
        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch = [im.convert("RGB") for im in images[i: i + batch_size]]
                if self.kind == "commfor":
                    x = torch.stack([self.processor(im) for im in batch])
                    out.append(torch.sigmoid(self.model(x)).reshape(-1).numpy())
                else:
                    inputs = self.processor(images=batch, return_tensors="pt")
                    probs = torch.softmax(self.model(**inputs).logits, dim=-1).numpy()
                    out.append(probs[:, self.ai_idx].sum(axis=1))
        return np.concatenate(out).astype(float)


def _load_commfor(repo: str):
    """Re-implementation of models.ViTClassifier from github.com/JeongsooP/Community-Forensics
    (timm ViT-S/16-384 with a 1-logit head) + its test transform."""
    import timm
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torchvision import transforms

    class ViTClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
            self.vit.head = torch.nn.Linear(384, 1)

        def forward(self, x):
            return self.vit(x)

    model = ViTClassifier()
    state = load_file(hf_hub_download(repo, "model.safetensors"))
    model.load_state_dict(state, strict=True)
    model.eval()
    tf = transforms.Compose([  # dataloader.get_transform(mode="test"), input_size=384
        transforms.Resize(440),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, tf


def load_detector(key: str) -> Detector:
    import torch

    torch.set_num_threads(max(1, torch.get_num_threads()))
    cfg = REGISTRY[key]
    t = time.time()
    if cfg["kind"] == "commfor":
        model, proc = _load_commfor(cfg["repo"])
        return Detector(key, cfg["repo"], cfg["desc"], model, proc, kind="commfor",
                        load_seconds=time.time() - t)

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    if "vit_processor" in cfg:
        from transformers import ViTImageProcessor

        proc = ViTImageProcessor(**cfg["vit_processor"])
    else:
        proc = AutoImageProcessor.from_pretrained(cfg["repo"])
    model = AutoModelForImageClassification.from_pretrained(cfg["repo"]).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    ai_idx = [i for i, lab in id2label.items() if lab in cfg["ai_labels"]]
    if not ai_idx:
        raise ValueError(f"{cfg['repo']}: none of {cfg['ai_labels']} in labels {id2label}")
    return Detector(key, cfg["repo"], cfg["desc"], model, proc, ai_idx, "hf", time.time() - t)
