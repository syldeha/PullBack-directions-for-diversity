"""Within-prompt CLIP, SSCD, and DINOv2 evaluation metrics."""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from configs import coco2017 as cfg
from generation import model


class PromptMetrics:
    """TPSO benchmark metrics plus DINOv2 similarity diagnostics."""

    def __init__(self):
        self.sscd = None
        self.clip_model = None
        self.clip_processor = None
        self.dino_model = None
        self.dino_transform = None

    def ensure_sscd(self):
        if self.sscd is not None:
            return
        checkpoint = Path(
            os.environ.get("SSCD_MODEL_PATH", cfg.SSCD_MODEL_PATH)
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "The SSCD metric checkpoint was not found. Set "
                f"SSCD_MODEL_PATH. Missing file: {checkpoint}"
            )
        self.sscd = torch.jit.load(
            str(checkpoint), map_location=model.device
        ).eval()

    def ensure_clip(self):
        if self.clip_model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self.clip_model = CLIPModel.from_pretrained(
            cfg.CLIP_MODEL_ID,
            local_files_only=cfg.METRIC_MODELS_LOCAL_ONLY,
        ).to(model.device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(
            cfg.CLIP_MODEL_ID,
            local_files_only=cfg.METRIC_MODELS_LOCAL_ONLY,
        )

    def ensure_dino(self):
        if self.dino_model is not None:
            return
        import timm
        from timm.data import create_transform, resolve_model_data_config

        self.dino_model = timm.create_model(
            cfg.DINO_MODEL_NAME,
            pretrained=True,
            num_classes=0,
        ).to(model.device).eval()
        for parameter in self.dino_model.parameters():
            parameter.requires_grad_(False)
        data_config = resolve_model_data_config(self.dino_model)
        self.dino_transform = create_transform(
            **data_config, is_training=False
        )

    def sscd_features(self, images):
        from torchvision.transforms import functional as tvf

        self.ensure_sscd()
        tensors = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for image in images:
            tensor = tvf.pil_to_tensor(
                image.convert("RGB").resize(
                    (320, 320), Image.Resampling.BICUBIC
                )
            ).float().div_(255.0)
            tensors.append(tvf.normalize(tensor, mean=mean, std=std))

        features = []
        with torch.no_grad():
            for start in range(0, len(tensors), cfg.METRIC_BATCH_SIZE):
                batch = torch.stack(
                    tensors[start:start + cfg.METRIC_BATCH_SIZE]
                ).to(model.device)
                features.append(self.sscd(batch).float().cpu())
        features = torch.cat(features)
        return features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def clip_score(self, images, prompt):
        self.ensure_clip()
        inputs = self.clip_processor(
            text=[prompt] * len(images),
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        with torch.no_grad():
            output = self.clip_model(**inputs)
            image = output.image_embeds.float()
            text = output.text_embeds.float()
        image = image / image.norm(dim=1, keepdim=True).clamp_min(1e-12)
        text = text / text.norm(dim=1, keepdim=True).clamp_min(1e-12)
        values = 100.0 * (image * text).sum(dim=1)
        return float(values.mean()), values.cpu().tolist()

    def dino_features(self, images):
        self.ensure_dino()
        tensors = [
            self.dino_transform(image.convert("RGB")) for image in images
        ]
        features = []
        with torch.no_grad():
            for start in range(0, len(tensors), cfg.METRIC_BATCH_SIZE):
                batch = torch.stack(
                    tensors[start:start + cfg.METRIC_BATCH_SIZE]
                ).to(model.device)
                output = self.dino_model(batch)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                features.append(output.float().cpu())
        features = torch.cat(features)
        return features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def pairwise_cosine_metrics(features):
        kernel = features @ features.T
        indexes = torch.triu_indices(len(features), len(features), offset=1)
        values = kernel[indexes[0], indexes[1]]
        if values.numel() == 0:
            raise ValueError("DINO metrics require at least two images")
        return {
            "dino_sim_mean": float(values.mean()),
            "dino_sim_max": float(values.max()),
            "dino_pairwise": values.tolist(),
        }

    def compute(self, images, prompt):
        sscd_features = self.sscd_features(images)
        kernel = sscd_features @ sscd_features.T

        # CADS includes diagonal self-similarities in MSS.
        mss = float(kernel.mean())
        eigenvalues = torch.linalg.eigvalsh(
            kernel / len(images)
        ).clamp_min(0)
        eigenvalues = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
        vendi = float(torch.exp(-(
            eigenvalues * (eigenvalues + 1e-12).log()
        ).sum()))
        clip, clip_all = self.clip_score(images, prompt)
        dino_features = self.dino_features(images)
        dino_metrics = self.pairwise_cosine_metrics(dino_features)
        return (
            {
                "clip": clip,
                "clip_all": clip_all,
                "mss": mss,
                "vendi": vendi,
                **dino_metrics,
            },
            {
                "sscd": sscd_features.numpy().astype(np.float32),
                "dino": dino_features.numpy().astype(np.float32),
            },
        )
