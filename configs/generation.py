"""Configuration for the canonical DDIM/CADS/TPSO/rho-star comparison.

Edit this file to choose captions, methods, pullback ranks, and evaluation
parameters. The runner is ``python -m experiments.generation``.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Pilot first. Increase to 250 only after inspecting the pilot summaries.
PROMPT_INDICES = list(range(500))
NUMBER_OF_PARTICLES = 5

# Main ablation axes. In disjoint mode, one particle receives approximately
# rank / NUMBER_OF_PARTICLES basis dimensions.
PULLBACK_RANKS = [40]
PULLBACK_POWER_ITERATIONS = [2]

# A maximum-rank basis is truncated so lower ranks are exact nested subspaces.
USE_NESTED_BASES = True

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
NEGATIVE_PROMPT = ""
HEIGHT = 512
WIDTH = 512
NUMBER_OF_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
ETA = 0.0
INITIAL_SEED = 12345
ETA_SEED = 20800
UNET_PARTICLE_BATCH_SIZE = 5
LOCAL_FILES_ONLY = True
REQUIRE_CUDA = True

# Pullback estimator and scheduled condition parameters.
PULLBACK_ANCHOR_TIMESTEP = 500
PULLBACK_INITIAL_SEED = 515
PULLBACK_SEED_PER_PROMPT = True
PULLBACK_FD_EPSILON = 0.5
PULLBACK_MODE = "disjoint"
PULLBACK_DIRECTION_SEED = 777
PULLBACK_START = 999
PULLBACK_END = 500
PULLBACK_SCHEDULE_POWER = 1.0

# Adaptive intermediate Jacobian refreshes.
PULLBACK_NUMBER_OF_REFRESHES = 2
PULLBACK_INTERMEDIATE_RANK = 20
PULLBACK_INTERMEDIATE_ITERATIONS = 1
PULLBACK_INTERMEDIATE_SEED = 1515
PULLBACK_TRANSITION_STEPS = 1

# CLIP-constrained, DINO-diversity rho-star selection. Zero is decoded as the
# clean reference but cannot be selected.
PROBE_TIMESTEP = 699
CANDIDATE_RHOS = [0.10, 0.125, 0.15, 0.175, 0.20, 0.25]
MAX_CLIP_DROP = 0.35
PROBE_DECODE_BATCH_SIZE = 4
SEARCH_STRATEGY = "beam"
MAX_SEARCH_COMBINATIONS = 5_000_000
SEARCH_BEAM_WIDTH = 4096
CONSTRAINT_FALLBACK = "minimum_selectable"

# Original CADS control used by the comparison.
CADS_START = 900
CADS_END = 600
CADS_NOISE_SCALE = 0.15
CADS_PSI = 1.0
CADS_NOISE_SEED = 999
CADS_PERSISTENCE = "fresh"
CADS_USE_RESCALE = True

# TPSO learns one positive-prompt token offset per particle. The text encoder
# and diffusion model stay frozen; the negative CFG branch stays clean.
TPSO_KAPPA = 0.80
TPSO_SIGMA = 0.01
TPSO_DIVERSITY_WEIGHT = 1.0
TPSO_LEARNING_RATE = 1e-3
TPSO_MAX_STEPS = 200
TPSO_MIN_STEPS = 50
TPSO_PATIENCE = 15
TPSO_MIN_DELTA = 1e-5
TPSO_INITIALIZATION_STD = 1e-4
TPSO_SEED = 3407
TPSO_RATIO = 0.40
TPSO_LOG_EVERY = 10

# Keep RUN_NAME unchanged to resume. Change it whenever any configuration
# value changes.
RUN_NAME = "rho_star_rank_iteration_ablation_coco20_vtpso_final"
OUTPUT_ROOT = ROOT / "outputs" / RUN_NAME
BASIS_CACHE = OUTPUT_ROOT / "basis_cache"
JPEG_QUALITY = 92

# Optional monitoring. CSV/JSON persistence does not depend on W&B.
USE_WANDB = True
WANDB_PROJECT = "sd15-pullback-rank-ablation"
WANDB_ENTITY = None
WANDB_MODE = os.environ.get("WANDB_MODE", "online")
WANDB_LOG_IMAGES = True


# Keep star imports in the runner limited to intentional configuration values.
__all__ = [name for name in globals() if name.isupper()]
