"""Editable configuration for the resumable BrushBench evaluation."""

import os

from inpainting.config import InpaintingConfig


# Keep this name to resume. Change it when any scientific parameter changes.
RUN_NAME = "brushbench_rho_star_final"

# Empty uses every locally available BrushBench image.
EXAMPLE_KEYS = []
MAX_EXAMPLES = 500
EXAMPLE_ORDER_SEED = 3407

SELECTED_METHODS = [
    "rho_star_pullback",
    "clean_ddim",
    "cads",
    "tpso",
]


# Rho-star pullback parameters for the BrushBench run.
EXPERIMENT = InpaintingConfig(
    num_particles=5,
    resolution=1024,
    ddim_steps=50,
    eta=0.0,
    initial_seed=4242,
    noise_seed_base=20800,

    # Regional adaptive disjoint pullback.
    pullback_rank=40,
    pullback_basis_timestep=600,
    pullback_basis_seed=515,
    pullback_basis_iterations=2,
    pullback_rho=2.5,
    pullback_start=999,
    pullback_end=500,
    pullback_schedule_power=2.0,
    pullback_direction_seed=777,
    pullback_refreshes=2,
    pullback_intermediate_rank=10,
    pullback_intermediate_iterations=1,
    pullback_intermediate_seed=1515,
    pullback_transition_steps=2,
    pullback_anchor_particle=0,
    pullback_response_region="edit_mask",  # "global" or "edit_mask"

    # Rho=0 is added automatically as the clean probe reference. It cannot be
    # selected unless it is explicitly added to this candidate list.
    rho_star_probe_timestep=699,
    rho_star_candidate_rhos=(1.0,1.25,1.5,1.75,2.0,2.5,3.0),
    rho_star_max_clip_drop=0.35,
    rho_star_search_strategy="beam",
    rho_star_max_combinations=5_000_000,
    rho_star_beam_width=4096,
    rho_star_constraint_fallback="minimum_selectable",
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
