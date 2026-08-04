"""Editable configuration for the resumable BrushBench evaluation."""

import os

from inpainting.config import InpaintingConfig


# Keep this name to resume. Change it when any scientific parameter changes.
RUN_NAME = "brushbench_comparison_v1"

# Empty uses every locally available BrushBench image.
EXAMPLE_KEYS = []
MAX_EXAMPLES = 250
EXAMPLE_ORDER_SEED = 3407

SELECTED_METHODS = [
    "clean_ddim",
    "cads",
    "adaptive_pullback",
    "tpso",
]


# These are the parameters from the audited long BrushBench comparison.
EXPERIMENT = InpaintingConfig(
    num_particles=4,
    resolution=1024,
    ddim_steps=50,
    eta=0.0,
    initial_seed=4242,
    noise_seed_base=20800,

    # Original CADS: fresh isotropic noise on both CFG branches.
    cads_noise_scale=0.15,
    cads_start=900,
    cads_end=600,
    cads_rescale_factor=1.0,
    cads_condition_seed=999,

    # Adaptive disjoint pullback, without rho-star selection.
    pullback_rank=40,
    pullback_basis_timestep=600,
    pullback_basis_seed=515,
    pullback_basis_iterations=2,
    pullback_rho=1.25,
    pullback_start=999,
    pullback_end=500,
    pullback_schedule_power=2.0,
    pullback_direction_seed=777,
    pullback_refreshes=2,
    pullback_intermediate_rank=8,
    pullback_intermediate_iterations=1,
    pullback_intermediate_seed=1515,
    pullback_transition_steps=2,
    pullback_anchor_particle=0,
    pullback_response_region="global",

    # TPSO token optimization and return-to-clean schedule.
    tpso_kappa=0.80,
    tpso_sigma=0.01,
    tpso_diversity_weight=1.0,
    tpso_learning_rate=1e-3,
    tpso_max_steps=200,
    tpso_min_steps=50,
    tpso_patience=15,
    tpso_initial_std=1e-4,
    tpso_seed=3407,
    tpso_ratio=0.4,
)


COMPUTE_MASK_CROP_METRICS = True
COMPUTE_RAW_OUTSIDE_PRESERVATION = True

SAVE_RAW_IMAGES = True
SAVE_COMPARISON_GRIDS = True
SHOW_SAMPLER_PROGRESS = True

USE_WANDB = True
WANDB_PROJECT = "pullback-inpainting-brushbench"
WANDB_ENTITY = None
WANDB_MODE = os.environ.get("WANDB_MODE", "online")
WANDB_LOG_GRIDS = True
WANDB_GRID_EVERY = 10
