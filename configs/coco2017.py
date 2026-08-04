"""Single configuration block for the MSCOCO 2017 TPSO-style benchmark."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------------
# TPSO evaluation protocol
# -----------------------------------------------------------------------------

# This repository preserves the 1,000-caption manifest used by the local
# experiments. Set the main count to 5,000 for the full TPSO paper protocol.
PROTOCOL = "main"
PROTOCOL_CAPTIONS = {"main": 1_000, "ablation": 1_000}
NUMBER_OF_CAPTIONS = PROTOCOL_CAPTIONS[PROTOCOL]
NUMBER_OF_PARTICLES = 5

# TPSO says "MSCOCO validation" but does not identify the year. We explicitly
# choose the modern 2017 validation split and record that choice in every run.
COCO_SPLIT = "val2017"
COCO_ROOT = Path(
    os.environ.get("COCO_ROOT", ROOT / "data" / "coco2017")
)
COCO_IMAGE_DIRECTORY = COCO_ROOT / "val2017"
COCO_CAPTIONS_FILE = COCO_ROOT / "annotations" / "captions_val2017.json"
CAPTION_SAMPLE_SEED = 2025
MANIFEST_DIRECTORY = COCO_ROOT / "manifests"
MANIFEST_PATH = MANIFEST_DIRECTORY / (
    f"{COCO_SPLIT}_captions_n{NUMBER_OF_CAPTIONS}_seed{CAPTION_SAMPLE_SEED}.jsonl"
)

# Keep unset for a reported benchmark. The environment override is used only
# by the automated one-caption integration test.
DEBUG_MAX_CAPTIONS = (
    int(os.environ["COCO_DEBUG_MAX_CAPTIONS"])
    if os.environ.get("COCO_DEBUG_MAX_CAPTIONS")
    else None
)

# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LOCAL_FILES_ONLY = True
REQUIRE_CUDA = True

SELECTED_METHODS = [
    "ddim_clean",
    "cads",
    "adaptive_pullback",
]

NEGATIVE_PROMPT = ""
HEIGHT = 512
WIDTH = 512
NUMBER_OF_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
ETA = 0.0

# Prompt p receives ten independent noises from this deterministic seed stream.
# Every method receives the exact same tensor, not merely the same distribution.
INITIAL_SEED = 12345
ETA_SEED = 20800
UNET_PARTICLE_BATCH_SIZE = 5

# Original CADS paper variant: fresh isotropic noise on both CFG branches,
# followed by branch-wise rescaling.
CADS_START = 900
CADS_END = 600
CADS_NOISE_SCALE = 0.15
CADS_PSI = 1.0
CADS_NOISE_SEED = 999
CADS_PERSISTENCE = "fresh"
CADS_USE_RESCALE = True

# Initial image-space pullback basis.
PULLBACK_ANCHOR_TIMESTEP = 500
PULLBACK_INITIAL_RANK = 32
PULLBACK_INITIAL_ITERATIONS = 3
PULLBACK_INITIAL_SEED = 515
PULLBACK_FD_EPSILON = 0.5

# Adaptive low-rank refreshes and interpolation.
PULLBACK_MODE = "disjoint"
PULLBACK_RHO = 0.15
PULLBACK_START = 999
PULLBACK_END = 350
PULLBACK_SCHEDULE_POWER = 1.0
PULLBACK_DIRECTION_SEED = 777
PULLBACK_NUMBER_OF_REFRESHES = 2
PULLBACK_INTERMEDIATE_RANK = 20
PULLBACK_INTERMEDIATE_ITERATIONS = 1
PULLBACK_INTERMEDIATE_SEED = 1515
PULLBACK_TRANSITION_STEPS = 4
PULLBACK_ANCHOR_PARTICLE = 0

# -----------------------------------------------------------------------------
# Storage, metrics, and W&B
# -----------------------------------------------------------------------------

RUN_NAME = os.environ.get(
    "COCO_RUN_NAME", f"sd15_coco2017_tpso_{PROTOCOL}_vfinal_1"
)
OUTPUT_ROOT = ROOT / "outputs" / RUN_NAME
BASIS_CACHE = OUTPUT_ROOT / "basis_cache"
JPEG_QUALITY = 90

SSCD_MODEL_PATH = Path(
    os.environ.get(
        "SSCD_MODEL_PATH",
        ROOT / "models" / "sscd_disc_mixup.torchscript.pt",
    )
)
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
# Same DINOv2 descriptor used by the earlier BrushNet experiments.
DINO_MODEL_NAME = "vit_base_patch14_dinov2"
METRIC_MODELS_LOCAL_ONLY = True
METRIC_BATCH_SIZE = 32

# Kynkaanniemi et al. use k-nearest-neighbour manifolds. This implementation
# records k explicitly because TPSO does not state it in the main paper.
PRECISION_RECALL_K = 3
GLOBAL_FEATURE_BATCH_SIZE = 64
DISTANCE_QUERY_BATCH_SIZE = 512
DISTANCE_REFERENCE_BATCH_SIZE = 4096

USE_WANDB = True
WANDB_PROJECT = "sd15-pullback-coco2017"
WANDB_ENTITY = None
WANDB_MODE = "online"
WANDB_LOG_IMAGE_EVERY = 100
