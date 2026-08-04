"""Metrics used by the BrushBench inpainting comparison."""

import itertools
import warnings

import numpy as np
import torch
from PIL import Image


class InpaintingMetrics:
    """LPIPS, DINOv2, CLIP, MSS, Vendi, and optional aesthetic metrics."""

    def __init__(self, device="cuda"):
        self.device = device
        self.lpips_model = None
        self.clip_model = None
        self.clip_processor = None
        self.dino_model = None
        self.dino_transform = None
        self.dino_unavailable = False
        self.aesthetic_model = None
        self.aesthetic_processor = None
        self.aesthetic_unavailable = False

    def load_lpips(self):
        if self.lpips_model is None:
            import lpips

            self.lpips_model = lpips.LPIPS(
                net="alex",
                verbose=False,
            ).to(self.device).eval()

    def load_clip(self):
        if self.clip_model is None:
            from transformers import CLIPModel, CLIPProcessor

            model_id = "openai/clip-vit-base-patch32"
            self.clip_model = CLIPModel.from_pretrained(model_id).to(
                self.device
            ).eval()
            self.clip_processor = CLIPProcessor.from_pretrained(model_id)

    def load_dino(self):
        if self.dino_model is not None or self.dino_unavailable:
            return
        try:
            import timm
            from timm.data import create_transform, resolve_model_data_config

            self.dino_model = timm.create_model(
                "vit_base_patch14_dinov2",
                pretrained=True,
                num_classes=0,
            ).to(self.device).eval()
            for parameter in self.dino_model.parameters():
                parameter.requires_grad_(False)
            data_config = resolve_model_data_config(self.dino_model)
            self.dino_transform = create_transform(
                **data_config,
                is_training=False,
            )
        except Exception as error:
            self.dino_model = None
            self.dino_transform = None
            self.dino_unavailable = True
            warnings.warn(
                "DINOv2 is unavailable; DINO metrics will be NaN. "
                f"Last error: {type(error).__name__}: {error}",
                RuntimeWarning,
            )

    def load_aesthetic(self):
        if self.aesthetic_model is not None or self.aesthetic_unavailable:
            return
        candidates = [
            "cafeai/cafe_aesthetic",
            "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE",
        ]
        last_error = None
        for model_id in candidates:
            try:
                from transformers import (
                    AutoImageProcessor,
                    AutoModelForImageClassification,
                )

                self.aesthetic_processor = AutoImageProcessor.from_pretrained(
                    model_id
                )
                self.aesthetic_model = (
                    AutoModelForImageClassification.from_pretrained(model_id)
                    .to(self.device)
                    .eval()
                )
                for parameter in self.aesthetic_model.parameters():
                    parameter.requires_grad_(False)
                return
            except Exception as error:
                last_error = error
                self.aesthetic_model = None
                self.aesthetic_processor = None
        self.aesthetic_unavailable = True
        warnings.warn(
            "The aesthetic model is unavailable; aesthetic metrics will be NaN. "
            f"Last error: {type(last_error).__name__}: {last_error}",
            RuntimeWarning,
        )

    @staticmethod
    def mask_bbox_crop(image, mask_image, padding_fraction=0.08, size=256):
        mask = np.asarray(mask_image) > 127
        if not mask.any():
            crop = image
        else:
            rows, columns = np.where(mask)
            top, bottom = rows.min(), rows.max() + 1
            left, right = columns.min(), columns.max() + 1
            vertical_padding = int((bottom - top) * padding_fraction)
            horizontal_padding = int((right - left) * padding_fraction)
            top = max(0, top - vertical_padding)
            bottom = min(image.height, bottom + vertical_padding)
            left = max(0, left - horizontal_padding)
            right = min(image.width, right + horizontal_padding)
            crop = image.crop((left, top, right, bottom))
        return crop.resize((size, size), Image.Resampling.LANCZOS)

    def pairwise_lpips(self, images, mask_image):
        self.load_lpips()
        crops = [self.mask_bbox_crop(image, mask_image) for image in images]
        tensors = torch.stack(
            [
                torch.from_numpy(np.asarray(crop, np.float32) / 127.5 - 1.0)
                .permute(2, 0, 1)
                for crop in crops
            ]
        ).to(self.device)
        values = []
        with torch.no_grad():
            for first, second in itertools.combinations(range(len(images)), 2):
                values.append(
                    float(
                        self.lpips_model(
                            tensors[first:first + 1],
                            tensors[second:second + 1],
                        ).item()
                    )
                )
        return values

    @staticmethod
    def pairwise_cosine_stats(embeddings):
        similarity = (embeddings @ embeddings.T).float().cpu()
        count = similarity.shape[0]
        if count < 2:
            return {"pairwise": [], "mean": float("nan"), "max": float("nan")}
        upper = torch.triu_indices(count, count, offset=1)
        values = similarity[upper[0], upper[1]]
        return {
            "pairwise": values.tolist(),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }

    def dino_embeddings(self, images, batch_size=5):
        self.load_dino()
        if self.dino_model is None:
            return None
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")

        outputs = []
        with torch.no_grad():
            for first in range(0, len(images), int(batch_size)):
                batch = images[first:first + int(batch_size)]
                tensors = torch.stack(
                    [self.dino_transform(image.convert("RGB")) for image in batch]
                ).to(self.device)
                embeddings = self.dino_model(tensors)
                if isinstance(embeddings, (list, tuple)):
                    embeddings = embeddings[0]
                embeddings = embeddings / embeddings.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
                outputs.append(embeddings.float().cpu())
        if not outputs:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.cat(outputs, dim=0)

    def dino_metrics(self, images):
        """Pairwise DINO cosine similarity; lower means more diversity."""
        embeddings = self.dino_embeddings(images)
        if embeddings is None:
            count = len(images) * (len(images) - 1) // 2
            return {
                "dino_pairwise": [float("nan")] * count,
                "dino_sim_mean": float("nan"),
                "dino_sim_max": float("nan"),
            }
        statistics = self.pairwise_cosine_stats(embeddings)
        return {
            "dino_pairwise": statistics["pairwise"],
            "dino_sim_mean": statistics["mean"],
            "dino_sim_max": statistics["max"],
        }

    def clip_metrics(self, images, caption):
        """Compute CLIPScore, MSS, and Vendi from one CLIP forward pass."""
        self.load_clip()
        with torch.no_grad():
            inputs = self.clip_processor(
                text=[caption],
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            output = self.clip_model(**inputs)
            image_embeddings = output.image_embeds / output.image_embeds.norm(
                dim=-1,
                keepdim=True,
            )
            text_embeddings = output.text_embeds / output.text_embeds.norm(
                dim=-1,
                keepdim=True,
            )
            clip_scores = (
                100.0 * image_embeddings @ text_embeddings.T
            ).squeeze(-1).cpu().tolist()

            similarity = (image_embeddings @ image_embeddings.T).float().cpu()
            count = similarity.shape[0]
            upper = torch.triu_indices(count, count, offset=1)
            mss = float(similarity[upper[0], upper[1]].mean())

            eigenvalues = torch.linalg.eigvalsh(similarity / count).clamp_min(0)
            eigenvalues = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
            vendi = float(
                torch.exp(
                    -(eigenvalues * (eigenvalues + 1e-12).log()).sum()
                )
            )
        return {"clip_all": clip_scores, "mss": mss, "vendi": vendi}

    def aesthetic_scores(self, images):
        self.load_aesthetic()
        if self.aesthetic_model is None:
            return [float("nan")] * len(images)
        with torch.no_grad():
            inputs = self.aesthetic_processor(
                images=images,
                return_tensors="pt",
            ).to(self.device)
            logits = self.aesthetic_model(**inputs).logits.float()
            if logits.shape[-1] == 1:
                values = logits.squeeze(-1)
            else:
                probabilities = logits.softmax(dim=-1)
                labels = getattr(self.aesthetic_model.config, "id2label", {})
                scores = []
                for index in range(logits.shape[-1]):
                    label = str(labels.get(index, index))
                    try:
                        scores.append(float(label))
                    except ValueError:
                        scores.append(float(index))
                score_tensor = torch.tensor(
                    scores,
                    device=probabilities.device,
                    dtype=probabilities.dtype,
                )
                values = probabilities @ score_tensor
        return values.cpu().tolist()

    def full_metrics(self, images, caption, mask_image):
        lpips_values = self.pairwise_lpips(images, mask_image)
        clip_values = self.clip_metrics(images, caption)
        dino_values = self.dino_metrics(images)
        aesthetic_values = self.aesthetic_scores(images)
        aesthetic_array = np.asarray(aesthetic_values, dtype=np.float32)
        aesthetic = (
            float(np.nanmean(aesthetic_array))
            if np.isfinite(aesthetic_array).any()
            else float("nan")
        )
        return {
            "lpips_mean": float(np.mean(lpips_values)),
            "lpips_min": float(np.min(lpips_values)),
            "lpips_all": lpips_values,
            "dino_sim_mean": dino_values["dino_sim_mean"],
            "dino_sim_max": dino_values["dino_sim_max"],
            "dino_pairwise": dino_values["dino_pairwise"],
            "vendi": clip_values["vendi"],
            "clip": float(np.mean(clip_values["clip_all"])),
            "clip_all": clip_values["clip_all"],
            "aesthetic": aesthetic,
            "aesthetic_all": aesthetic_values,
            "mss": clip_values["mss"],
        }
