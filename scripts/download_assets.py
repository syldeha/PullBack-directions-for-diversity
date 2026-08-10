"""Download the model assets for one documented workflow profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from scripts.workflow_profiles import (
    CLIP_REPOSITORY,
    DINO_REPOSITORY,
    PROFILE_NAMES,
    SD15_REPOSITORY,
    SDXL_REPOSITORY,
    get_profile,
)


BASE_ALLOW_PATTERNS = {
    "sd15": [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ],
    "sdxl": [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "tokenizer_2/*",
        "text_encoder/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ],
}

ADAPTER_FILES = {
    "sd15": {
        "config.json": "1z8jex6js-zzSxURnNOWMW1PP4njvtusE",
        "diffusion_pytorch_model.safetensors": "1uLQ55rdcFljdP_9iYGX5ZmJWzXFJast7",
    },
    "sdxl": {
        "config.json": "1jCt6KsP2jKSBD_QYJsuQ9Yb1rujLLt3f",
        "diffusion_pytorch_model.safetensors": "1yPhfQ3y60fXURr9_YfQrdE48y68IZjBN",
    },
}


def require_compatible_base_target(profile, target):
    model_index = target / "model_index.json"
    if not model_index.is_file():
        return
    try:
        pipeline_class = json.loads(model_index.read_text()).get("_class_name")
    except Exception as error:
        raise RuntimeError(f"Cannot read existing {model_index}: {error}") from error
    expected = (
        "StableDiffusionPipeline"
        if profile.family == "sd15"
        else "StableDiffusionXLPipeline"
    )
    if pipeline_class != expected:
        raise RuntimeError(
            f"Refusing to download {profile.family} into {target}: existing "
            f"model is {pipeline_class}, expected {expected}. Choose a "
            "different --base-model directory."
        )


def require_compatible_adapter_target(profile, target):
    config_path = target / "config.json"
    if not config_path.is_file():
        return
    try:
        dimension = json.loads(config_path.read_text()).get("cross_attention_dim")
    except Exception as error:
        raise RuntimeError(f"Cannot read existing {config_path}: {error}") from error
    expected = 768 if profile.family == "sd15" else 2048
    if dimension != expected:
        raise RuntimeError(
            f"Refusing to download {profile.family} adapter into {target}: "
            f"existing cross_attention_dim={dimension}, expected {expected}. "
            "Choose a different --brushnet-model directory."
        )


def download_base(profile, force=False, dry_run=False):
    repository = SD15_REPOSITORY if profile.family == "sd15" else SDXL_REPOSITORY
    target = Path(profile.base_model).expanduser().resolve()
    print(f"base model: {repository} -> {target}")
    require_compatible_base_target(profile, target)
    if dry_run:
        return
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repository,
        local_dir=str(target),
        allow_patterns=BASE_ALLOW_PATTERNS[profile.family],
        force_download=force,
    )


def download_adapter(profile, force=False, dry_run=False):
    if profile.task != "inpainting":
        return
    import gdown

    target = profile.brushnet_model
    print(f"BrushNet adapter: Google Drive -> {target}")
    require_compatible_adapter_target(profile, target)
    if dry_run:
        return
    target.mkdir(parents=True, exist_ok=True)
    for filename, file_id in ADAPTER_FILES[profile.family].items():
        destination = target / filename
        if destination.is_file() and not force:
            print(f"  keeping existing {destination}")
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        if temporary.exists():
            temporary.unlink()
        result = gdown.download(
            id=file_id,
            output=str(temporary),
            quiet=False,
        )
        if not result or not temporary.is_file():
            raise RuntimeError(f"Google Drive download failed for {filename}")
        temporary.replace(destination)


def download_metrics(dry_run=False):
    for repository in (CLIP_REPOSITORY, DINO_REPOSITORY):
        print(f"metric model: {repository} -> Hugging Face cache")
        if not dry_run:
            snapshot_download(repo_id=repository)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--brushnet-model")
    parser.add_argument("--include-metrics", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    profile = get_profile(
        args.profile,
        base_model=args.base_model,
        brushnet_model=args.brushnet_model,
    )
    try:
        download_base(profile, force=args.force, dry_run=args.dry_run)
        download_adapter(profile, force=args.force, dry_run=args.dry_run)
        if args.include_metrics:
            download_metrics(dry_run=args.dry_run)
    except RuntimeError as error:
        parser.exit(2, f"ERROR: {error}\n")
    if profile.task == "inpainting":
        print(
            "BrushBench is license-controlled and is not downloaded by this "
            "script. Set BRUSHBENCH_ROOT after following the upstream steps."
        )


if __name__ == "__main__":
    main()
