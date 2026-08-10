"""One source of truth for public generation and inpainting profiles."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BRUSHNET_COMMIT = "0f9d9e54ca85c40a11a8f0504b4b5b2e7e8fd14d"
SD15_REPOSITORY = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SDXL_REPOSITORY = "stabilityai/stable-diffusion-xl-base-1.0"
CLIP_REPOSITORY = "openai/clip-vit-base-patch32"
DINO_REPOSITORY = "timm/vit_base_patch14_dinov2.lvd142m"


@dataclass(frozen=True)
class WorkflowProfile:
    name: str
    task: str
    family: str
    resolution: int
    base_model: str
    brushnet_root: Path
    brushnet_model: Path | None
    brushbench_root: Path | None


PROFILE_NAMES = (
    "generation-sd15",
    "generation-sdxl",
    "inpainting-sd15",
    "inpainting-sdxl",
)


def brushnet_root():
    return Path(
        os.environ.get("BRUSHNET_ROOT", Path.home() / "BrushNet")
    ).expanduser().resolve()


def default_base_model(family, root=None):
    root = root or brushnet_root()
    folder = (
        "stable-diffusion-v1-5"
        if family == "sd15"
        else "stable-diffusion-xl-base-1.0"
    )
    override = os.environ.get(f"PULLBACK_{family.upper()}_MODEL")
    return str(Path(override).expanduser()) if override else str(root / "checkpoints" / folder)


def get_profile(name, base_model=None, brushnet_model=None, brushbench_root=None):
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile {name!r}; choose from {PROFILE_NAMES}")
    task, family = name.split("-", maxsplit=1)
    root = brushnet_root()
    base = (
        str(Path(base_model).expanduser())
        if base_model
        else default_base_model(family, root)
    )
    adapter = None
    dataset = None
    if task == "inpainting":
        default_adapter = root / "checkpoints" / (
            "brushnet_segmentation_mask" if family == "sd15" else "brushnet_sdxl"
        )
        adapter = Path(
            brushnet_model or default_adapter
        ).expanduser().resolve()
        default_dataset = root / "data" / "BrushBench"
        dataset = Path(
            brushbench_root
            or os.environ.get("BRUSHBENCH_ROOT", default_dataset)
        ).expanduser().resolve()
    return WorkflowProfile(
        name=name,
        task=task,
        family=family,
        resolution=512 if family == "sd15" else 1024,
        base_model=base,
        brushnet_root=root,
        brushnet_model=adapter,
        brushbench_root=dataset,
    )


def configure_inpainting_environment(profile):
    if profile.task != "inpainting":
        return
    os.environ["BRUSHNET_ROOT"] = str(profile.brushnet_root)
    os.environ["BRUSHNET_MODEL_FAMILY"] = profile.family
    os.environ["BRUSHNET_BASE_MODEL"] = profile.base_model
    os.environ["BRUSHNET_MODEL"] = str(profile.brushnet_model)
    os.environ["BRUSHBENCH_ROOT"] = str(profile.brushbench_root)
