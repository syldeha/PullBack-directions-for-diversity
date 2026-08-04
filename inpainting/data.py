"""BrushBench example loading."""

from dataclasses import dataclass
from pathlib import Path
import json
import os

import numpy as np
from PIL import Image

from inpainting import model


BRUSHBENCH_ROOT = Path(
    os.environ.get(
        "BRUSHBENCH_ROOT",
        model.BRUSHNET_ROOT / "data" / "BrushBench",
    )
)


@dataclass
class BrushBenchExample:
    key: str
    source_image: Image.Image
    mask: np.ndarray
    prompt: str

    @property
    def mask_fraction(self):
        return float(self.mask.mean())


def brushbench_path():
    """Resolve the dataset path again so notebook path edits take effect."""
    return Path(os.environ.get("BRUSHBENCH_ROOT", BRUSHBENCH_ROOT))


def available_keys():
    """Return metadata keys whose source images are available locally."""
    root = brushbench_path()
    with (root / "mapping_file.json").open() as handle:
        mapping = json.load(handle)
    return [
        key
        for key, item in mapping.items()
        if (root / item["image"]).exists()
    ]


def load_example(key):
    """Load one source image, official mask, and BrushBench caption."""
    root = brushbench_path()
    key = str(key)
    if key.isdigit():
        key = f"{int(key):09d}"

    with (root / "mapping_file.json").open() as handle:
        mapping = json.load(handle)
    if key not in mapping:
        raise KeyError(f"BrushBench key {key!r} does not exist")

    item = mapping[key]
    image_path = root / item["image"]
    if not image_path.exists():
        preview = ", ".join(available_keys()[:12])
        raise FileNotFoundError(
            f"{image_path.name} is not downloaded. Available keys begin with: "
            f"{preview}"
        )
    source = Image.open(image_path).convert("RGB")
    mask = model.decode_rle_mask(
        item["inpainting_mask"],
        (source.height, source.width),
    )
    return BrushBenchExample(key, source, mask, item["caption"])
