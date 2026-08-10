"""Ablate pullback rank and block-power iterations for rho-star sampling.

Every (rank, iterations) pair is treated as a separate rho-star method. Clean
DDIM, paper CADS, TPSO, the initial latent batch, and the clean probe prefix are
shared within a caption.

Edit ``configs/generation.py``, then run:

    conda run --no-capture-output -n pullback \
        python -u -m experiments.generation

The run is resumable per (caption, method). Results and aggregate CSV files are
updated after every caption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from configs import coco2017 as coco_cfg
from configs.generation import *  # noqa: F403 - experiment parameters
from evaluation.metrics import PromptMetrics
from generation import ddim, model
from methods import cads, rho_star as rs, tpso
from methods.adaptive_pullback import sample_adaptive_pullback
from pullback import basis as pullback_basis
from pullback import directions


# =============================================================================
# Configuration helpers
# =============================================================================

BASELINE_METHODS = ["clean_ddim", "cads", "tpso"]

# These keys are the only additions allowed when resuming a run created before
# TPSO was added. All original experiment parameters must still match exactly.
TPSO_EXTENSION_KEYS = {
    "baseline_methods",
    "tpso_kappa",
    "tpso_sigma",
    "tpso_diversity_weight",
    "tpso_learning_rate",
    "tpso_max_steps",
    "tpso_min_steps",
    "tpso_patience",
    "tpso_min_delta",
    "tpso_initialization_std",
    "tpso_seed",
    "tpso_ratio",
    "tpso_log_every",
}


def ablation_configs():
    configs = []
    for iterations in sorted(PULLBACK_POWER_ITERATIONS):
        for rank in sorted(PULLBACK_RANKS):
            configs.append({
                "method": f"rho_star_rank{rank}_iter{iterations}",
                "rank": int(rank),
                "iterations": int(iterations),
            })
    return configs


def method_names():
    return BASELINE_METHODS + [config["method"] for config in ablation_configs()]


def method_config(method):
    for config in ablation_configs():
        if config["method"] == method:
            return config
    return {"method": method, "rank": None, "iterations": None}


def configuration():
    return {
        "baseline_methods": BASELINE_METHODS,
        "prompt_indices": PROMPT_INDICES,
        "number_of_particles": NUMBER_OF_PARTICLES,
        "pullback_ranks": PULLBACK_RANKS,
        "pullback_power_iterations": PULLBACK_POWER_ITERATIONS,
        "use_nested_bases": USE_NESTED_BASES,
        "model_family": MODEL_FAMILY,
        "model_id": MODEL_ID,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": HEIGHT,
        "width": WIDTH,
        "number_of_ddim_steps": NUMBER_OF_DDIM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "eta": ETA,
        "initial_seed": INITIAL_SEED,
        "eta_seed": ETA_SEED,
        "unet_particle_batch_size": UNET_PARTICLE_BATCH_SIZE,
        "pullback_anchor_timestep": PULLBACK_ANCHOR_TIMESTEP,
        "pullback_initial_seed": PULLBACK_INITIAL_SEED,
        "pullback_seed_per_prompt": PULLBACK_SEED_PER_PROMPT,
        "pullback_fd_epsilon": PULLBACK_FD_EPSILON,
        "pullback_mode": PULLBACK_MODE,
        "pullback_direction_seed": PULLBACK_DIRECTION_SEED,
        "pullback_start": PULLBACK_START,
        "pullback_end": PULLBACK_END,
        "pullback_schedule_power": PULLBACK_SCHEDULE_POWER,
        "pullback_number_of_refreshes": PULLBACK_NUMBER_OF_REFRESHES,
        "pullback_intermediate_rank": PULLBACK_INTERMEDIATE_RANK,
        "pullback_intermediate_iterations": PULLBACK_INTERMEDIATE_ITERATIONS,
        "pullback_intermediate_seed": PULLBACK_INTERMEDIATE_SEED,
        "pullback_transition_steps": PULLBACK_TRANSITION_STEPS,
        "probe_timestep": PROBE_TIMESTEP,
        "candidate_rhos": CANDIDATE_RHOS,
        "max_clip_drop": MAX_CLIP_DROP,
        "probe_decode_batch_size": PROBE_DECODE_BATCH_SIZE,
        "search_strategy": SEARCH_STRATEGY,
        "max_search_combinations": MAX_SEARCH_COMBINATIONS,
        "search_beam_width": SEARCH_BEAM_WIDTH,
        "constraint_fallback": CONSTRAINT_FALLBACK,
        "cads_start": CADS_START,
        "cads_end": CADS_END,
        "cads_noise_scale": CADS_NOISE_SCALE,
        "cads_psi": CADS_PSI,
        "cads_noise_seed": CADS_NOISE_SEED,
        "cads_persistence": CADS_PERSISTENCE,
        "cads_use_rescale": CADS_USE_RESCALE,
        "tpso_kappa": TPSO_KAPPA,
        "tpso_sigma": TPSO_SIGMA,
        "tpso_diversity_weight": TPSO_DIVERSITY_WEIGHT,
        "tpso_learning_rate": TPSO_LEARNING_RATE,
        "tpso_max_steps": TPSO_MAX_STEPS,
        "tpso_min_steps": TPSO_MIN_STEPS,
        "tpso_patience": TPSO_PATIENCE,
        "tpso_min_delta": TPSO_MIN_DELTA,
        "tpso_initialization_std": TPSO_INITIALIZATION_STD,
        "tpso_seed": TPSO_SEED,
        "tpso_ratio": TPSO_RATIO,
        "tpso_log_every": TPSO_LOG_EVERY,
        "manifest_path": str(coco_cfg.MANIFEST_PATH),
    }


def is_tpso_only_configuration_extension(previous, requested):
    """Allow an old compatible run to gain TPSO without rerunning baselines."""

    previous_keys = set(previous)
    requested_keys = set(requested)
    added_keys = requested_keys - previous_keys
    if not previous_keys.issubset(requested_keys):
        return False
    if not added_keys or not added_keys.issubset(TPSO_EXTENSION_KEYS):
        return False
    return all(requested[key] == value for key, value in previous.items())


def validate_configuration():
    if not PROMPT_INDICES:
        raise ValueError("PROMPT_INDICES cannot be empty")
    if len(set(PROMPT_INDICES)) != len(PROMPT_INDICES):
        raise ValueError("PROMPT_INDICES must be unique")
    if NUMBER_OF_PARTICLES < 2:
        raise ValueError("At least two particles are required")
    if not PULLBACK_RANKS or not PULLBACK_POWER_ITERATIONS:
        raise ValueError("The rank and iteration lists cannot be empty")
    if len(set(PULLBACK_RANKS)) != len(PULLBACK_RANKS):
        raise ValueError("PULLBACK_RANKS must be unique")
    if len(set(PULLBACK_POWER_ITERATIONS)) != len(PULLBACK_POWER_ITERATIONS):
        raise ValueError("PULLBACK_POWER_ITERATIONS must be unique")
    if any(int(rank) < NUMBER_OF_PARTICLES for rank in PULLBACK_RANKS):
        raise ValueError("Every disjoint rank must be >= NUMBER_OF_PARTICLES")
    if any(int(value) < 1 for value in PULLBACK_POWER_ITERATIONS):
        raise ValueError("Power iteration counts must be positive")
    if not USE_NESTED_BASES:
        raise ValueError(
            "This controlled runner currently requires USE_NESTED_BASES=True"
        )
    if PULLBACK_MODE != "disjoint":
        raise ValueError("This ablation is designed for disjoint pullback")
    if PULLBACK_NUMBER_OF_REFRESHES > 0:
        if PULLBACK_INTERMEDIATE_RANK < NUMBER_OF_PARTICLES:
            raise ValueError("Intermediate rank must be >= particles")
        if PULLBACK_INTERMEDIATE_ITERATIONS < 1:
            raise ValueError("Intermediate iterations must be positive")
    if not CANDIDATE_RHOS:
        raise ValueError("CANDIDATE_RHOS cannot be empty")
    if len(set(float(value) for value in CANDIDATE_RHOS)) != len(CANDIDATE_RHOS):
        raise ValueError("CANDIDATE_RHOS must be unique")
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in CANDIDATE_RHOS
    ):
        raise ValueError("Selectable rhos must be finite and strictly positive")
    if MAX_CLIP_DROP < 0.0 or not math.isfinite(float(MAX_CLIP_DROP)):
        raise ValueError("MAX_CLIP_DROP must be finite and non-negative")
    if SEARCH_STRATEGY not in {"auto", "exact", "beam"}:
        raise ValueError("SEARCH_STRATEGY must be auto, exact, or beam")
    combinations = len(CANDIDATE_RHOS) ** NUMBER_OF_PARTICLES
    if SEARCH_STRATEGY == "exact" and combinations > MAX_SEARCH_COMBINATIONS:
        raise ValueError(
            f"Exact rho search has {combinations:,} combinations; use beam"
        )
    if CONSTRAINT_FALLBACK not in {"clean", "minimum_selectable", "error"}:
        raise ValueError("Unknown CONSTRAINT_FALLBACK")
    if CADS_PERSISTENCE not in {"fresh", "fixed"}:
        raise ValueError("CADS_PERSISTENCE must be fresh or fixed")
    if CADS_START <= CADS_END:
        raise ValueError("CADS_START must be greater than CADS_END")
    if not 0.0 < TPSO_KAPPA <= 1.0:
        raise ValueError("TPSO_KAPPA must be in (0, 1]")
    if TPSO_SIGMA < 0.0:
        raise ValueError("TPSO_SIGMA must be non-negative")
    if TPSO_DIVERSITY_WEIGHT < 0.0:
        raise ValueError("TPSO_DIVERSITY_WEIGHT must be non-negative")
    if TPSO_LEARNING_RATE <= 0.0:
        raise ValueError("TPSO_LEARNING_RATE must be positive")
    if not 1 <= TPSO_MIN_STEPS <= TPSO_MAX_STEPS:
        raise ValueError(
            "TPSO steps must satisfy 1 <= MIN_STEPS <= MAX_STEPS"
        )
    if TPSO_PATIENCE < 1:
        raise ValueError("TPSO_PATIENCE must be positive")
    if TPSO_MIN_DELTA < 0.0:
        raise ValueError("TPSO_MIN_DELTA must be non-negative")
    if TPSO_INITIALIZATION_STD < 0.0:
        raise ValueError("TPSO_INITIALIZATION_STD must be non-negative")
    if not 0.0 < TPSO_RATIO <= 1.0:
        raise ValueError("TPSO_RATIO must be in (0, 1]")
    if not coco_cfg.MANIFEST_PATH.exists():
        raise FileNotFoundError(coco_cfg.MANIFEST_PATH)


def load_selected_manifest():
    with coco_cfg.MANIFEST_PATH.open() as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]
    if max(PROMPT_INDICES) >= len(manifest):
        raise ValueError("A selected prompt index is outside the manifest")
    return [manifest[index] for index in PROMPT_INDICES]


# =============================================================================
# Persistence and timing
# =============================================================================

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    temporary.replace(path)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(function):
    synchronize()
    started = time.perf_counter()
    value = function()
    synchronize()
    return value, time.perf_counter() - started


def method_paths(prompt_index: int, method: str):
    root = OUTPUT_ROOT / f"c{prompt_index:05d}" / method
    return {
        "root": root,
        "result": root / "result.json",
        "sscd": root / "sscd.npy",
        "dino": root / "dino.npy",
        "images": [
            root / f"p{particle:02d}.jpg"
            for particle in range(NUMBER_OF_PARTICLES)
        ],
    }


def load_completed(prompt_index: int, method: str):
    paths = method_paths(prompt_index, method)
    required = [
        paths["result"], paths["sscd"], paths["dino"], *paths["images"]
    ]
    if not all(path.exists() for path in required):
        return None
    with paths["result"].open() as handle:
        result = json.load(handle)
    images = [Image.open(path).convert("RGB") for path in paths["images"]]
    return result, images


def save_method_result(row, method, images, metrics, features, metadata):
    paths = method_paths(row["prompt_index"], method)
    paths["root"].mkdir(parents=True, exist_ok=True)
    for image, path in zip(images, paths["images"]):
        image.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    np.save(paths["sscd"], features["sscd"])
    np.save(paths["dino"], features["dino"])
    result = {
        **row,
        "method": method,
        "metrics": metrics,
        "metadata": metadata,
    }
    write_json(paths["result"], result)
    return result


# =============================================================================
# Controlled nested basis family
# =============================================================================

def basis_estimator_seed(row):
    if PULLBACK_SEED_PER_PROMPT:
        return PULLBACK_INITIAL_SEED + row["prompt_index"]
    return PULLBACK_INITIAL_SEED


def basis_family_filename(row):
    identity = {
        "prompt_index": row["prompt_index"],
        "caption": row["caption"],
        "anchor": PULLBACK_ANCHOR_TIMESTEP,
        "max_rank": max(PULLBACK_RANKS),
        "max_iterations": max(PULLBACK_POWER_ITERATIONS),
        "seed": basis_estimator_seed(row),
        "fd_epsilon": PULLBACK_FD_EPSILON,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:16]
    return f"c{row['prompt_index']:05d}_{digest}.pt"


def rayleigh_ritz_basis(q, metric_q):
    q_flat = q.flatten(start_dim=1)
    metric_flat = metric_q.flatten(start_dim=1)
    projected = q_flat @ metric_flat.T
    projected = 0.5 * (projected + projected.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(projected)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    basis_flat = eigenvectors.T @ q_flat
    basis = basis_flat.reshape(-1, *q.shape[1:])
    basis = pullback_basis.orthonormalize_directions(basis)
    return basis, eigenvalues.detach().float()


def compute_nested_basis_snapshots(
    latent,
    timestep,
    positive_condition,
    number_of_real_tokens,
    max_rank,
    max_iterations,
    seed,
    progress_label,
):
    """Return ordered max-rank bases after every power iteration."""

    center = positive_condition[:, :number_of_real_tokens, :].detach().clone()
    condition_shape = center.shape[1:]
    max_rank = min(int(max_rank), int(center[0].numel()))

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    directions = torch.randn(
        (max_rank, *condition_shape),
        generator=generator,
        dtype=torch.float32,
    ).to(model.device)
    q = pullback_basis.orthonormalize_directions(directions)

    print(f"{progress_label}: initial metric block")
    metric_q = pullback_basis.apply_pullback_metric_block(
        latent,
        timestep,
        center,
        q,
        positive_condition,
        number_of_real_tokens,
        PULLBACK_FD_EPSILON,
    )

    snapshots = {}
    for iteration in range(1, int(max_iterations) + 1):
        print(
            f"{progress_label}: power iteration "
            f"{iteration}/{max_iterations}"
        )
        q = pullback_basis.orthonormalize_directions(metric_q)
        metric_q = pullback_basis.apply_pullback_metric_block(
            latent,
            timestep,
            center,
            q,
            positive_condition,
            number_of_real_tokens,
            PULLBACK_FD_EPSILON,
        )
        basis, eigenvalues = rayleigh_ritz_basis(q, metric_q)
        snapshots[str(iteration)] = {
            "basis": basis.detach().cpu().to(torch.float16),
            "eigenvalues": eigenvalues.cpu(),
            "verification": pullback_basis.verify_basis(basis),
        }
    return snapshots


def get_basis_family(
    row,
    initial_latents,
    positive,
    negative,
    number_of_real_tokens,
):
    cache_path = BASIS_CACHE / basis_family_filename(row)
    if cache_path.exists():
        pack = torch.load(cache_path, map_location="cpu", weights_only=True)
        return pack, "local"

    anchor_pack, anchor_seconds = timed_call(
        lambda: directions.compute_clean_anchor(
            initial_latent=initial_latents[0:1],
            positive_condition=positive,
            negative_condition=negative,
            number_of_steps=NUMBER_OF_DDIM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            eta=ETA,
            eta_seed=ETA_SEED + row["prompt_index"],
            requested_timestep=PULLBACK_ANCHOR_TIMESTEP,
        )
    )
    anchor, anchor_timestep, _ = anchor_pack
    snapshots, spectral_seconds = timed_call(
        lambda: compute_nested_basis_snapshots(
            latent=anchor,
            timestep=anchor_timestep,
            positive_condition=positive,
            number_of_real_tokens=number_of_real_tokens,
            max_rank=max(PULLBACK_RANKS),
            max_iterations=max(PULLBACK_POWER_ITERATIONS),
            seed=basis_estimator_seed(row),
            progress_label=f"basis family c{row['prompt_index']:05d}",
        )
    )
    pack = {
        "snapshots": snapshots,
        "anchor_timestep": int(anchor_timestep.item()),
        "max_rank": int(max(PULLBACK_RANKS)),
        "max_iterations": int(max(PULLBACK_POWER_ITERATIONS)),
        "estimator_seed": int(basis_estimator_seed(row)),
        "timing_seconds": {
            "anchor": anchor_seconds,
            "spectral_family": spectral_seconds,
            "total": anchor_seconds + spectral_seconds,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, cache_path)
    return pack, "computed"


def basis_for_config(family, rank, iterations):
    snapshot = family["snapshots"][str(int(iterations))]
    basis = snapshot["basis"].to(model.device, dtype=torch.float32)
    if basis.shape[0] < int(rank):
        raise RuntimeError(
            f"Basis contains {basis.shape[0]} directions, requested {rank}"
        )
    # The cache is FP16 to keep the full rank-by-iteration sweep compact.
    # Restore numerical orthogonality before disjoint projection.
    basis = pullback_basis.orthonormalize_directions(
        basis[:int(rank)].contiguous()
    )
    if basis.shape[0] != int(rank):
        raise RuntimeError(
            f"Cached basis lost rank: restored {basis.shape[0]}, requested {rank}"
        )
    eigenvalues = snapshot["eigenvalues"][:int(rank)].float().tolist()
    return basis, eigenvalues


def estimated_direct_basis_seconds(family, rank, iterations):
    """Estimate standalone cost from the exact number of metric matvecs."""

    timing = family["timing_seconds"]
    numerator = float(rank) * (float(iterations) + 1.0)
    denominator = float(family["max_rank"]) * (
        float(family["max_iterations"]) + 1.0
    )
    return timing["anchor"] + timing["spectral_family"] * numerator / denominator


# =============================================================================
# Summaries, grids, and W&B
# =============================================================================

METRIC_NAMES = ["clip", "mss", "vendi", "dino_sim_mean", "dino_sim_max"]


def rebuild_summaries(manifest):
    metric_rows = []
    timing_rows = []
    selection_rows = []
    particle_rho_rows = []

    for row in manifest:
        for method in method_names():
            completed = load_completed(row["prompt_index"], method)
            if completed is None:
                continue
            result, _ = completed
            config = method_config(method)
            common = {
                "prompt_index": row["prompt_index"],
                "method": method,
                "rank": config["rank"],
                "power_iterations": config["iterations"],
            }
            metric_rows.append({
                **common,
                **{name: result["metrics"][name] for name in METRIC_NAMES},
            })
            timing_rows.append({
                **common,
                **result["metadata"].get("timing_seconds", {}),
            })

            if config["rank"] is None:
                continue
            selected_rhos = np.asarray(
                result["metadata"]["selected_rhos"], dtype=np.float64
            )
            selection = result["metadata"]["selection"]
            selection_rows.append({
                **common,
                "directions_per_particle_min": min(
                    result["metadata"]["basis"]["direction_counts"]
                ),
                "directions_per_particle_max": max(
                    result["metadata"]["basis"]["direction_counts"]
                ),
                "selected_rho_mean": float(selected_rhos.mean()),
                "selected_rho_std": float(selected_rhos.std()),
                "selected_rho_min": float(selected_rhos.min()),
                "selected_rho_max": float(selected_rhos.max()),
                "probe_dino_diversity_gain": selection[
                    "dino_diversity_gain"
                ],
                "probe_clip_change_mean": selection[
                    "selected_clip_change_mean"
                ],
                "probe_clip_change_min": selection[
                    "selected_clip_change_min"
                ],
                "constraint_fallback_count": len(selection.get(
                    "constraint_fallback_particles", []
                )),
            })
            for particle, rho in enumerate(selected_rhos.tolist()):
                particle_rho_rows.append({
                    **common,
                    "particle": particle,
                    "selected_rho": rho,
                })

    metrics_frame = pd.DataFrame(metric_rows)
    timing_frame = pd.DataFrame(timing_rows)
    selection_frame = pd.DataFrame(selection_rows)
    particle_rho_frame = pd.DataFrame(particle_rho_rows)

    if not metrics_frame.empty:
        metrics_frame.to_csv(OUTPUT_ROOT / "per_prompt_metrics.csv", index=False)
        summary = metrics_frame.groupby("method")[METRIC_NAMES].agg(
            ["mean", "std", "min", "max"]
        )
        summary.columns = [
            f"{metric}_{statistic}" for metric, statistic in summary.columns
        ]
        summary = summary.reset_index()
        config_frame = pd.DataFrame([
            method_config(method) for method in summary["method"]
        ]).drop(columns="method")
        summary = pd.concat([summary[["method"]], config_frame, summary.drop(
            columns="method"
        )], axis=1)
        summary.to_csv(OUTPUT_ROOT / "metrics_summary.csv", index=False)
        summary_records = (
            summary.astype(object)
            .where(pd.notnull(summary), None)
            .to_dict(orient="records")
        )
        write_json(
            OUTPUT_ROOT / "metrics_summary.json",
            summary_records,
        )
        visible = [
            "method", "clip_mean", "dino_sim_mean_mean",
            "dino_sim_max_mean", "mss_mean", "vendi_mean",
        ]
        print("\nCurrent metric summary:")
        print(summary[visible].round(4).to_string(index=False))

    if not timing_frame.empty:
        timing_frame.to_csv(OUTPUT_ROOT / "per_prompt_timing.csv", index=False)
        numeric = [
            column for column in timing_frame.select_dtypes(
                include=["number"]
            ).columns
            if column not in {"prompt_index", "rank", "power_iterations"}
        ]
        timing_summary = timing_frame.groupby("method")[numeric].agg(
            ["mean", "std", "min", "max"]
        )
        timing_summary.columns = [
            f"{metric}_{statistic}"
            for metric, statistic in timing_summary.columns
        ]
        timing_summary.reset_index().to_csv(
            OUTPUT_ROOT / "timing_summary.csv", index=False
        )

    if not selection_frame.empty:
        selection_frame.to_csv(
            OUTPUT_ROOT / "per_prompt_selection.csv", index=False
        )
        numeric = [
            column for column in selection_frame.select_dtypes(
                include=["number"]
            ).columns
            if column not in {"prompt_index", "rank", "power_iterations"}
        ]
        selection_summary = selection_frame.groupby("method")[numeric].agg(
            ["mean", "std", "min", "max"]
        )
        selection_summary.columns = [
            f"{metric}_{statistic}"
            for metric, statistic in selection_summary.columns
        ]
        selection_summary.reset_index().to_csv(
            OUTPUT_ROOT / "selection_summary.csv", index=False
        )

    if not particle_rho_frame.empty:
        particle_rho_frame.to_csv(
            OUTPUT_ROOT / "per_particle_selected_rho.csv", index=False
        )
        counts = particle_rho_frame.groupby(
            ["method", "rank", "power_iterations", "selected_rho"]
        ).size().rename("count").reset_index()
        totals = counts.groupby("method")["count"].transform("sum")
        counts["fraction"] = counts["count"] / totals
        counts.to_csv(
            OUTPUT_ROOT / "selected_rho_distribution.csv", index=False
        )


def make_method_strip(method, images):
    thumb = 256
    title_height = 34
    canvas = Image.new(
        "RGB",
        (thumb * NUMBER_OF_PARTICLES, thumb + title_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for particle, image in enumerate(images):
        x = particle * thumb
        draw.text((x + 5, 7), f"{method} p{particle}", fill="black")
        resized = image.resize((thumb, thumb), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x, title_height))
    return canvas


def make_comparison_grid(caption, images_by_method):
    thumb = 256
    title_height = 34
    caption_height = 46
    ordered = [
        method for method in method_names() if method in images_by_method
    ]
    canvas = Image.new(
        "RGB",
        (
            thumb * NUMBER_OF_PARTICLES,
            caption_height + len(ordered) * (thumb + title_height),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), caption[:180], fill="black")
    for row_index, method in enumerate(ordered):
        y = caption_height + row_index * (thumb + title_height)
        for particle, image in enumerate(images_by_method[method]):
            x = particle * thumb
            draw.text((x + 5, y + 7), f"{method} p{particle}", fill="black")
            resized = image.resize((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(resized, (x, y + title_height))
    return canvas


def initialize_wandb():
    if not USE_WANDB:
        return None
    try:
        import wandb

        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=RUN_NAME,
            id=RUN_NAME,
            resume="allow",
            mode=WANDB_MODE,
            config=configuration(),
        )
        wandb.define_metric("prompt_order")
        wandb.define_metric("metrics/*", step_metric="prompt_order")
        wandb.define_metric("selection/*", step_metric="prompt_order")
        wandb.define_metric("timing/*", step_metric="prompt_order")
        return run
    except Exception as error:
        print("WARNING: W&B disabled:", type(error).__name__, error)
        return None


def log_method_to_wandb(
    run,
    prompt_order,
    prompt_index,
    method,
    metrics,
    metadata,
    images,
):
    if run is None:
        return
    import wandb

    payload = {
        "prompt_order": prompt_order,
        "prompt_index": prompt_index,
    }
    for metric in METRIC_NAMES:
        payload[f"metrics/{metric}/{method}"] = metrics[metric]
    timing = metadata.get("timing_seconds", {})
    for name in (
        "optimization", "sampling", "total_generation",
        "basis_estimated_direct", "selection_overhead", "total_estimated_direct",
    ):
        if name in timing:
            payload[f"timing/{name}/{method}"] = timing[name]
    if "selected_rhos" in metadata:
        selected = np.asarray(metadata["selected_rhos"], dtype=np.float64)
        payload[f"selection/rho_mean/{method}"] = float(selected.mean())
        payload[f"selection/fallbacks/{method}"] = len(
            metadata["selection"].get("constraint_fallback_particles", [])
        )
    if WANDB_LOG_IMAGES:
        payload[f"images/{method}"] = wandb.Image(
            make_method_strip(method, images),
            caption=f"c{prompt_index:05d}: {method}",
        )
    run.log(payload)


# =============================================================================
# Main experiment
# =============================================================================

def main(check_only=False):
    validate_configuration()
    manifest = load_selected_manifest()

    if check_only:
        print(
            f"run={RUN_NAME} captions={len(manifest)} "
            f"particles={NUMBER_OF_PARTICLES} methods={method_names()}"
        )
        print("model:", MODEL_FAMILY, MODEL_ID)
        print("manifest:", coco_cfg.MANIFEST_PATH)
        print("output:", OUTPUT_ROOT)
        print("configuration is valid; model weights were not loaded")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    config_path = OUTPUT_ROOT / "config.json"
    requested_config = configuration()
    if config_path.exists():
        with config_path.open() as handle:
            previous_config = json.load(handle)
        if previous_config != requested_config:
            if is_tpso_only_configuration_extension(
                previous_config, requested_config
            ):
                backup_path = OUTPUT_ROOT / "config_before_tpso.json"
                if not backup_path.exists():
                    write_json(backup_path, previous_config)
                write_json(config_path, requested_config)
                print(
                    "Extended the existing compatible run with TPSO config; "
                    "completed methods will be reused."
                )
            else:
                raise RuntimeError(
                    f"Configuration changed for {OUTPUT_ROOT}. Change RUN_NAME."
                )
    else:
        write_json(config_path, requested_config)

    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")

    print(
        f"rank/iteration ablation: captions={len(manifest)} "
        f"particles={NUMBER_OF_PARTICLES} ranks={PULLBACK_RANKS} "
        f"iterations={PULLBACK_POWER_ITERATIONS}"
    )
    model.load_model(MODEL_ID, family=MODEL_FAMILY, local_files_only=LOCAL_FILES_ONLY)
    model.set_unet_particle_batch_size(UNET_PARTICLE_BATCH_SIZE)
    metrics_calculator = PromptMetrics()
    wandb_run = initialize_wandb()

    for prompt_order, row in enumerate(manifest):
        prompt_index = row["prompt_index"]
        caption = row["caption"]
        print(
            f"\n===== caption {prompt_order + 1}/{len(manifest)} "
            f"index={prompt_index} ====="
        )
        print(caption)

        positive, negative, number_of_real_tokens = model.encode_prompt(
            caption, NEGATIVE_PROMPT
        )
        initial_latents = model.make_initial_latents(
            NUMBER_OF_PARTICLES,
            HEIGHT,
            WIDTH,
            INITIAL_SEED + prompt_index,
        )
        latent_hash = hashlib.sha256(
            initial_latents.detach().cpu().numpy().tobytes()
        ).hexdigest()
        images_by_method = {}

        clean_completed = load_completed(prompt_index, "clean_ddim")
        if clean_completed is None:
            print("running: clean_ddim")
            clean_latents, clean_sampling_seconds = timed_call(
                lambda: ddim.sample_clean_ddim(
                    initial_latents=initial_latents,
                    positive_condition=positive,
                    negative_condition=negative,
                    number_of_steps=NUMBER_OF_DDIM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    eta=ETA,
                    eta_seed=ETA_SEED + prompt_index,
                    progress=True,
                )
            )
            clean_images, clean_decode_seconds = timed_call(
                lambda: model.decode_latents(clean_latents)
            )
            (clean_metrics, clean_features), clean_metric_seconds = timed_call(
                lambda: metrics_calculator.compute(clean_images, caption)
            )
            clean_metadata = {
                "initial_latent_sha256": latent_hash,
                "timing_seconds": {
                    "sampling": clean_sampling_seconds,
                    "decode": clean_decode_seconds,
                    "metrics": clean_metric_seconds,
                },
            }
            save_method_result(
                row,
                "clean_ddim",
                clean_images,
                clean_metrics,
                clean_features,
                clean_metadata,
            )
            log_method_to_wandb(
                wandb_run,
                prompt_order,
                prompt_index,
                "clean_ddim",
                clean_metrics,
                clean_metadata,
                clean_images,
            )
            del clean_latents
            print(
                f"clean_ddim CLIP={clean_metrics['clip']:.3f} "
                f"DINO={clean_metrics['dino_sim_mean']:.4f} "
                f"MSS={clean_metrics['mss']:.4f} "
                f"Vendi={clean_metrics['vendi']:.3f}"
            )
        else:
            clean_result, clean_images = clean_completed
            if clean_result["metadata"]["initial_latent_sha256"] != latent_hash:
                raise RuntimeError("Saved clean result has different initial noise")
            clean_sampling_seconds = clean_result["metadata"][
                "timing_seconds"
            ]["sampling"]
            print("loaded: clean_ddim")
        images_by_method["clean_ddim"] = clean_images

        cads_completed = load_completed(prompt_index, "cads")
        if cads_completed is None:
            print("running: cads")
            cads_latents, cads_sampling_seconds = timed_call(
                lambda: cads.sample_cads(
                    initial_latents=initial_latents,
                    positive_condition=positive,
                    negative_condition=negative,
                    number_of_steps=NUMBER_OF_DDIM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    eta=ETA,
                    eta_seed=ETA_SEED + prompt_index,
                    start=CADS_START,
                    end=CADS_END,
                    noise_scale=CADS_NOISE_SCALE,
                    psi=CADS_PSI,
                    noise_seed=CADS_NOISE_SEED + prompt_index,
                    persistence=CADS_PERSISTENCE,
                    use_rescale=CADS_USE_RESCALE,
                    progress=True,
                )
            )
            cads_images, cads_decode_seconds = timed_call(
                lambda: model.decode_latents(cads_latents)
            )
            (cads_metrics, cads_features), cads_metric_seconds = timed_call(
                lambda: metrics_calculator.compute(cads_images, caption)
            )
            cads_metadata = {
                "initial_latent_sha256": latent_hash,
                "paper_parameters": {
                    "start": CADS_START,
                    "end": CADS_END,
                    "noise_scale": CADS_NOISE_SCALE,
                    "psi": CADS_PSI,
                    "noise_seed": CADS_NOISE_SEED + prompt_index,
                    "persistence": CADS_PERSISTENCE,
                    "use_rescale": CADS_USE_RESCALE,
                },
                "timing_seconds": {
                    "sampling": cads_sampling_seconds,
                    "decode": cads_decode_seconds,
                    "metrics": cads_metric_seconds,
                },
            }
            save_method_result(
                row,
                "cads",
                cads_images,
                cads_metrics,
                cads_features,
                cads_metadata,
            )
            log_method_to_wandb(
                wandb_run,
                prompt_order,
                prompt_index,
                "cads",
                cads_metrics,
                cads_metadata,
                cads_images,
            )
            del cads_latents
            print(
                f"cads CLIP={cads_metrics['clip']:.3f} "
                f"DINO={cads_metrics['dino_sim_mean']:.4f} "
                f"MSS={cads_metrics['mss']:.4f} "
                f"Vendi={cads_metrics['vendi']:.3f}"
            )
        else:
            cads_result, cads_images = cads_completed
            if cads_result["metadata"]["initial_latent_sha256"] != latent_hash:
                raise RuntimeError("Saved CADS result has different initial noise")
            print("loaded: cads")
        images_by_method["cads"] = cads_images

        tpso_completed = load_completed(prompt_index, "tpso")
        if tpso_completed is None:
            print("running: tpso")
            zero_offset_verification = tpso.verify_clean_encoding(
                caption, positive
            )
            if (
                zero_offset_verification["positive_condition_max_error"]
                > 1e-5
            ):
                raise RuntimeError(
                    "TPSO zero-offset encoding does not match clean SD1.5"
                )
            optimized, tpso_optimization_seconds = timed_call(
                lambda: tpso.optimize_token_offsets(
                    prompt=caption,
                    number_of_particles=NUMBER_OF_PARTICLES,
                    kappa=TPSO_KAPPA,
                    sigma=TPSO_SIGMA,
                    diversity_weight=TPSO_DIVERSITY_WEIGHT,
                    learning_rate=TPSO_LEARNING_RATE,
                    max_steps=TPSO_MAX_STEPS,
                    min_steps=TPSO_MIN_STEPS,
                    patience=TPSO_PATIENCE,
                    min_delta=TPSO_MIN_DELTA,
                    initialization_std=TPSO_INITIALIZATION_STD,
                    seed=TPSO_SEED + prompt_index,
                    log_every=TPSO_LOG_EVERY,
                )
            )
            tpso_sample_pack, tpso_sampling_seconds = timed_call(
                lambda: tpso.sample_tpso(
                    initial_latents=initial_latents,
                    clean_positive=positive,
                    negative_condition=negative,
                    optimized=optimized,
                    number_of_steps=NUMBER_OF_DDIM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    eta=ETA,
                    eta_seed=ETA_SEED + prompt_index,
                    ratio=TPSO_RATIO,
                    progress=True,
                )
            )
            tpso_latents, alpha_trace = tpso_sample_pack
            tpso_images, tpso_decode_seconds = timed_call(
                lambda: model.decode_latents(tpso_latents)
            )
            (tpso_metrics, tpso_features), tpso_metric_seconds = timed_call(
                lambda: metrics_calculator.compute(tpso_images, caption)
            )
            tpso_metadata = {
                "initial_latent_sha256": latent_hash,
                "optimization": {
                    "kappa": TPSO_KAPPA,
                    "sigma": TPSO_SIGMA,
                    "diversity_weight": TPSO_DIVERSITY_WEIGHT,
                    "learning_rate": TPSO_LEARNING_RATE,
                    "max_steps": TPSO_MAX_STEPS,
                    "min_steps": TPSO_MIN_STEPS,
                    "patience": TPSO_PATIENCE,
                    "min_delta": TPSO_MIN_DELTA,
                    "initialization_std": TPSO_INITIALIZATION_STD,
                    "seed": TPSO_SEED + prompt_index,
                    "steps_run": optimized.steps_run,
                    "offset_norms": optimized.offset_norms,
                    "final_diagnostics": optimized.final_diagnostics,
                    "history_tail": optimized.history[-5:],
                    "zero_offset_verification": zero_offset_verification,
                },
                "schedule": {
                    "ratio": TPSO_RATIO,
                    "positive_branch": "optimized_to_clean",
                    "negative_branch": "clean",
                    "alpha_trace": [
                        {"timestep": timestep, "alpha": alpha}
                        for timestep, alpha in alpha_trace
                    ],
                },
                "timing_seconds": {
                    "optimization": tpso_optimization_seconds,
                    "optimization_internal": optimized.optimization_seconds,
                    "sampling": tpso_sampling_seconds,
                    "total_generation": (
                        tpso_optimization_seconds + tpso_sampling_seconds
                    ),
                    "additional_vs_clean": (
                        tpso_optimization_seconds
                        + tpso_sampling_seconds
                        - clean_sampling_seconds
                    ),
                    "decode": tpso_decode_seconds,
                    "metrics": tpso_metric_seconds,
                },
            }
            save_method_result(
                row,
                "tpso",
                tpso_images,
                tpso_metrics,
                tpso_features,
                tpso_metadata,
            )
            log_method_to_wandb(
                wandb_run,
                prompt_order,
                prompt_index,
                "tpso",
                tpso_metrics,
                tpso_metadata,
                tpso_images,
            )
            del tpso_latents, optimized
            print(
                f"tpso CLIP={tpso_metrics['clip']:.3f} "
                f"DINO={tpso_metrics['dino_sim_mean']:.4f} "
                f"MSS={tpso_metrics['mss']:.4f} "
                f"Vendi={tpso_metrics['vendi']:.3f} "
                f"opt={tpso_optimization_seconds:.2f}s "
                f"sample={tpso_sampling_seconds:.2f}s"
            )
        else:
            tpso_result, tpso_images = tpso_completed
            if tpso_result["metadata"]["initial_latent_sha256"] != latent_hash:
                raise RuntimeError("Saved TPSO result has different initial noise")
            print("loaded: tpso")
        images_by_method["tpso"] = tpso_images

        incomplete_configs = [
            config for config in ablation_configs()
            if load_completed(prompt_index, config["method"]) is None
        ]
        family = None
        family_source = None
        family_access_seconds = 0.0
        probe_latents = None
        actual_probe_timestep = None
        probe_step_index = None
        probe_prefix_seconds = 0.0

        if incomplete_configs:
            print("building/loading controlled basis family")
            family_pack, family_access_seconds = timed_call(
                lambda: get_basis_family(
                    row,
                    initial_latents,
                    positive,
                    negative,
                    number_of_real_tokens,
                )
            )
            family, family_source = family_pack

            print(f"shared clean probe -> requested t={PROBE_TIMESTEP}")
            probe_pack, probe_prefix_seconds = timed_call(
                lambda: directions.compute_clean_anchor(
                    initial_latent=initial_latents,
                    positive_condition=positive,
                    negative_condition=negative,
                    number_of_steps=NUMBER_OF_DDIM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    eta=ETA,
                    eta_seed=ETA_SEED + prompt_index,
                    requested_timestep=PROBE_TIMESTEP,
                )
            )
            probe_latents, actual_probe_timestep, probe_step_index = probe_pack

        for config in ablation_configs():
            method = config["method"]
            completed = load_completed(prompt_index, method)
            if completed is not None:
                result, images = completed
                if result["metadata"]["initial_latent_sha256"] != latent_hash:
                    raise RuntimeError(
                        f"Saved {method} result has different initial noise"
                    )
                images_by_method[method] = images
                print("loaded:", method)
                continue

            rank = config["rank"]
            iterations = config["iterations"]
            print(f"running: {method}")
            basis, eigenvalues = basis_for_config(family, rank, iterations)
            direction_counts = [
                len(indexes)
                for indexes in directions.snake_slices(
                    rank,
                    NUMBER_OF_PARTICLES,
                )
            ]
            full_scale_directions = rs.make_full_scale_directions(
                basis=basis,
                positive_condition=positive,
                number_of_real_tokens=number_of_real_tokens,
                number_of_particles=NUMBER_OF_PARTICLES,
                mode=PULLBACK_MODE,
                direction_seed=PULLBACK_DIRECTION_SEED,
            )

            probe_rhos = [0.0] + [float(value) for value in CANDIDATE_RHOS]
            probe, candidate_probe_seconds = timed_call(
                lambda: rs.probe_rho_candidates(
                    latents=probe_latents,
                    timestep=actual_probe_timestep,
                    positive_condition=positive,
                    negative_condition=negative,
                    number_of_real_tokens=number_of_real_tokens,
                    full_scale_directions=full_scale_directions,
                    candidate_rhos=probe_rhos,
                    guidance_scale=GUIDANCE_SCALE,
                    schedule_start=PULLBACK_START,
                    schedule_end=PULLBACK_END,
                    schedule_power=PULLBACK_SCHEDULE_POWER,
                )
            )
            probe, probe_feature_seconds = timed_call(
                lambda: rs.add_clip_dino_probe_features(
                    probe=probe,
                    prompt=caption,
                    metrics_calculator=metrics_calculator,
                    decode_batch_size=PROBE_DECODE_BATCH_SIZE,
                )
            )
            selection, search_seconds = timed_call(
                lambda: rs.select_rho_combination_clip_dino(
                    probe=probe,
                    max_clip_drop=MAX_CLIP_DROP,
                    max_combinations=MAX_SEARCH_COMBINATIONS,
                    selectable_rhos=CANDIDATE_RHOS,
                    search_strategy=SEARCH_STRATEGY,
                    beam_width=SEARCH_BEAM_WIDTH,
                    constraint_fallback=CONSTRAINT_FALLBACK,
                )
            )
            selected_rhos = selection["selected_rhos"]
            print("selected rhos:", [round(value, 4) for value in selected_rhos])

            candidate_rows = rs.candidate_diagnostic_rows(probe, selection)
            candidate_path = (
                OUTPUT_ROOT
                / f"c{prompt_index:05d}"
                / method
                / "rho_candidates.csv"
            )
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(candidate_rows).to_csv(candidate_path, index=False)

            sample_pack, sampling_seconds = timed_call(
                lambda: sample_adaptive_pullback(
                    initial_latents=initial_latents,
                    positive_condition=positive,
                    negative_condition=negative,
                    number_of_real_tokens=number_of_real_tokens,
                    initial_basis=basis,
                    number_of_steps=NUMBER_OF_DDIM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    eta=ETA,
                    eta_seed=ETA_SEED + prompt_index,
                    rho=0.0,
                    particle_rho=selected_rhos,
                    start=PULLBACK_START,
                    end=PULLBACK_END,
                    schedule_power=PULLBACK_SCHEDULE_POWER,
                    mode=PULLBACK_MODE,
                    direction_seed=PULLBACK_DIRECTION_SEED,
                    number_of_refreshes=PULLBACK_NUMBER_OF_REFRESHES,
                    intermediate_rank=PULLBACK_INTERMEDIATE_RANK,
                    intermediate_iterations=PULLBACK_INTERMEDIATE_ITERATIONS,
                    intermediate_seed=PULLBACK_INTERMEDIATE_SEED,
                    transition_steps=PULLBACK_TRANSITION_STEPS,
                    finite_difference_epsilon=PULLBACK_FD_EPSILON,
                    progress=True,
                )
            )
            latents, sampler_metadata = sample_pack
            images, decode_seconds = timed_call(
                lambda: model.decode_latents(latents)
            )
            (metrics, features), metric_seconds = timed_call(
                lambda: metrics_calculator.compute(images, caption)
            )

            basis_direct_seconds = estimated_direct_basis_seconds(
                family, rank, iterations
            )
            selection_overhead = (
                probe_prefix_seconds
                + candidate_probe_seconds
                + probe_feature_seconds
                + search_seconds
            )
            total_direct = basis_direct_seconds + selection_overhead + sampling_seconds
            timing = {
                "sampling": sampling_seconds,
                "probe_clean_prefix": probe_prefix_seconds,
                "probe_candidates": candidate_probe_seconds,
                "probe_clip_dino_features": probe_feature_seconds,
                "rho_search": search_seconds,
                "selection_overhead": selection_overhead,
                "basis_family_access": family_access_seconds,
                "basis_family_original_total": family["timing_seconds"]["total"],
                "basis_estimated_direct": basis_direct_seconds,
                "total_estimated_direct": total_direct,
                "additional_vs_clean_estimated_direct": (
                    total_direct - clean_sampling_seconds
                ),
                "decode": decode_seconds,
                "metrics": metric_seconds,
            }
            metadata = {
                "initial_latent_sha256": latent_hash,
                "rank": rank,
                "power_iterations": iterations,
                "selected_rhos": selected_rhos,
                "selection": {
                    key: value for key, value in selection.items()
                    if key not in {"feasible_mask", "constraint_feasible_mask"}
                },
                "probe": {
                    "requested_timestep": PROBE_TIMESTEP,
                    "actual_timestep": int(actual_probe_timestep.item()),
                    "clean_prefix_steps": int(probe_step_index),
                    "schedule_envelope": probe["schedule_envelope"],
                    "probe_rhos": probe_rhos,
                    "selectable_rhos": CANDIDATE_RHOS,
                    "max_clip_drop": MAX_CLIP_DROP,
                    "selection_fidelity": "one-sided CLIP drop from rho=0",
                    "selection_diversity": "mean pairwise DINO cosine distance",
                },
                "basis": {
                    "family_source": family_source,
                    "nested_max_rank": family["max_rank"],
                    "nested_max_iterations": family["max_iterations"],
                    "rank": rank,
                    "power_iterations": iterations,
                    "direction_counts": direction_counts,
                    "anchor_timestep": family["anchor_timestep"],
                    "estimator_seed": family["estimator_seed"],
                    "eigenvalues": eigenvalues,
                    "verification": pullback_basis.verify_basis(basis),
                },
                "sampler": sampler_metadata,
                "timing_seconds": timing,
            }
            save_method_result(
                row, method, images, metrics, features, metadata
            )
            log_method_to_wandb(
                wandb_run,
                prompt_order,
                prompt_index,
                method,
                metrics,
                metadata,
                images,
            )
            images_by_method[method] = images
            print(
                f"{method} CLIP={metrics['clip']:.3f} "
                f"DINO={metrics['dino_sim_mean']:.4f} "
                f"MSS={metrics['mss']:.4f} "
                f"Vendi={metrics['vendi']:.3f} "
                f"sample={sampling_seconds:.2f}s "
                f"basis_est={basis_direct_seconds:.2f}s"
            )
            del latents, probe, basis, full_scale_directions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if probe_latents is not None:
            del probe_latents

        grid = make_comparison_grid(caption, images_by_method)
        grid_path = OUTPUT_ROOT / f"c{prompt_index:05d}" / "comparison_grid.jpg"
        grid.save(grid_path, quality=JPEG_QUALITY)
        if wandb_run is not None and WANDB_LOG_IMAGES:
            import wandb

            wandb_run.log({
                "prompt_order": prompt_order,
                "prompt_index": prompt_index,
                "comparison/grid": wandb.Image(grid, caption=caption),
            })

        rebuild_summaries(manifest)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rebuild_summaries(manifest)
    if wandb_run is not None:
        wandb_run.finish()
    print("\nablation complete:", OUTPUT_ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate configuration and data without loading model weights",
    )
    arguments = parser.parse_args()
    main(check_only=arguments.check)
