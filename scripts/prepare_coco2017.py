"""Download COCO 2017 validation data and build the fixed caption manifest."""

from __future__ import annotations

import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

from configs import coco2017 as cfg


VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


def download(url: str, destination: Path):
    if destination.exists():
        print("using existing:", destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print("downloading:", url)
    with urllib.request.urlopen(url) as source, partial.open("wb") as target:
        shutil.copyfileobj(source, target)
    partial.replace(destination)


def extract(zip_path: Path, target: Path, required_path: Path):
    if required_path.exists():
        print("using extracted:", required_path)
        return
    print("extracting:", zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)
    if not required_path.exists():
        raise FileNotFoundError(f"Archive did not create {required_path}")


def build_manifest(number_of_captions: int, seed: int, output_path: Path):
    with cfg.COCO_CAPTIONS_FILE.open() as handle:
        annotations = json.load(handle)["annotations"]

    # Sorting makes the seeded shuffle stable even if JSON ordering changes.
    annotations = sorted(annotations, key=lambda row: int(row["id"]))
    random.Random(seed).shuffle(annotations)
    selected = annotations[:number_of_captions]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for prompt_index, annotation in enumerate(selected):
            image_id = int(annotation["image_id"])
            row = {
                "prompt_index": prompt_index,
                "annotation_id": int(annotation["id"]),
                "image_id": image_id,
                "caption": annotation["caption"].strip(),
                "real_image": f"{image_id:012d}.jpg",
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    temporary.replace(output_path)
    print("wrote manifest:", output_path)
    print("captions:", len(selected))
    print("unique reference images:", len({row["image_id"] for row in selected}))


def validate_manifest(path: Path, expected_count: int):
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != expected_count:
        raise RuntimeError(f"Expected {expected_count} rows, found {len(rows)}")
    for expected_index, row in enumerate(rows):
        if row["prompt_index"] != expected_index:
            raise RuntimeError("Manifest prompt indices are not contiguous")
        image = cfg.COCO_IMAGE_DIRECTORY / row["real_image"]
        if not image.exists():
            raise FileNotFoundError(image)
    print("manifest validation passed:", len(rows), "captions")


def main():
    cfg.COCO_ROOT.mkdir(parents=True, exist_ok=True)
    images_zip = cfg.COCO_ROOT / "val2017.zip"
    annotations_zip = cfg.COCO_ROOT / "annotations_trainval2017.zip"

    required_image = cfg.COCO_IMAGE_DIRECTORY / "000000000139.jpg"
    if not required_image.exists():
        download(VAL_IMAGES_URL, images_zip)
        extract(images_zip, cfg.COCO_ROOT, required_image)
    else:
        print("using extracted:", required_image)

    if not cfg.COCO_CAPTIONS_FILE.exists():
        download(ANNOTATIONS_URL, annotations_zip)
        extract(annotations_zip, cfg.COCO_ROOT, cfg.COCO_CAPTIONS_FILE)
    else:
        print("using extracted:", cfg.COCO_CAPTIONS_FILE)
    build_manifest(
        cfg.NUMBER_OF_CAPTIONS,
        cfg.CAPTION_SAMPLE_SEED,
        cfg.MANIFEST_PATH,
    )
    validate_manifest(cfg.MANIFEST_PATH, cfg.NUMBER_OF_CAPTIONS)


if __name__ == "__main__":
    main()
