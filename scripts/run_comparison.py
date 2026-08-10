"""Run a compact method comparison and save individual images plus a grid."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np

from scripts.workflow_profiles import (
    PROFILE_NAMES,
    REPOSITORY_ROOT,
    configure_inpainting_environment,
    get_profile,
)


METHODS = ("clean_ddim", "cads", "tpso", "rho_star")
METHOD_LABELS = {
    "clean_ddim": "Clean DDIM",
    "cads": "CADS",
    "tpso": "TPSO",
    "rho_star": "Pullback (ours)",
}


# ---------------------------------------------------------------------------
# Reproducible comparison configuration
# ---------------------------------------------------------------------------
#
# Keep every scientific parameter used by this compact comparison visible in
# this block. Command-line arguments only override the example/prompt, number
# of DDIM steps, number of particles, model paths, and output directory.

DEFAULT_EXAMPLE_KEY = "000000089"
DEFAULT_PROMPT = "A red fox standing in an autumn forest, photorealistic"
NEGATIVE_PROMPT = ""
DEFAULT_DDIM_STEPS = 10
DEFAULT_NUM_PARTICLES = 4
MINIMUM_DDIM_STEPS = 10
MINIMUM_NUM_PARTICLES = 2

# Rho magnitudes are model-family dependent. Rho=0 is added automatically as
# the clean probe reference and is not part of the selectable candidate tuple.
MODEL_SPECIFIC_PARAMETERS = {
    "sd15": {
        "rho_star_candidate_rhos": (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
    },
    "sdxl": {
        "rho_star_candidate_rhos": (1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
    },
}

# Shared DDIM sampling parameters.
GUIDANCE_SCALE = 7.5
ETA = 0.0
GENERATION_INITIAL_SEED = 12345
INPAINTING_INITIAL_SEED = 4242
NOISE_SEED_BASE = 20800
UNET_PARTICLE_BATCH_SIZE = 1
INPAINTING_COMPARISON_RESOLUTION_CAP = 512

# Pullback basis, directions, and scheduled perturbation.
PULLBACK_MINIMUM_RANK = 2
PULLBACK_BASIS_TIMESTEP = 600
PULLBACK_BASIS_SEED = 515
PULLBACK_BASIS_ITERATIONS = 1
PULLBACK_FINITE_DIFFERENCE_EPSILON = 0.5
PULLBACK_MODE = "disjoint"
PULLBACK_DIRECTION_SEED = 777
PULLBACK_START = 999
PULLBACK_END = 500
PULLBACK_SCHEDULE_POWER = 2.0
PULLBACK_NUMBER_OF_REFRESHES = 1
PULLBACK_INTERMEDIATE_ITERATIONS = 1
PULLBACK_INTERMEDIATE_SEED = 1515
PULLBACK_TRANSITION_STEPS = 1
PULLBACK_ANCHOR_PARTICLE = 0
PULLBACK_RESPONSE_REGION = "edit_mask"

# CLIP-constrained, DINO-diversity rho-star selection.
RHO_STAR_PROBE_TIMESTEP = 699
GENERATION_RHO_STAR_MAX_CLIP_DROP = 0.5
INPAINTING_RHO_STAR_MAX_CLIP_DROP = 0.35
RHO_STAR_SEARCH_STRATEGY = "exact"
RHO_STAR_MAX_COMBINATIONS = 5_000_000
RHO_STAR_BEAM_WIDTH = 4096
RHO_STAR_CONSTRAINT_FALLBACK = "minimum_selectable"
RHO_STAR_PROBE_DECODE_BATCH_SIZE = 2

# CADS baseline.
CADS_START = 900
CADS_END = 600
CADS_NOISE_SCALE = 0.15
CADS_RESCALE_FACTOR = 1.0
CADS_PSI = 1.0
CADS_CONDITION_SEED = 999
CADS_PERSISTENCE = "fresh"
CADS_USE_RESCALE = True

# TPSO baseline. These intentionally small optimization counts keep this
# public comparison compact; the long paper configurations use larger values.
TPSO_KAPPA = 0.80
TPSO_SIGMA = 0.01
TPSO_DIVERSITY_WEIGHT = 1.0
TPSO_LEARNING_RATE = 1e-3
TPSO_MAX_STEPS = 4
TPSO_MIN_STEPS = 2
TPSO_PATIENCE = 2
TPSO_INITIALIZATION_STD = 1e-4
TPSO_SEED = 3407
TPSO_RATIO = 0.4
TPSO_LOG_EVERY = 1


def parse_methods(value):
    if value == "all":
        return list(METHODS)
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown methods {unknown}; choose from {METHODS} or 'all'"
        )
    if not methods:
        raise argparse.ArgumentTypeError("at least one method is required")
    return methods


def validate_and_save(images_by_method, output_directory, expected_count):
    records = {}
    for method, images in images_by_method.items():
        method_directory = output_directory / method
        method_directory.mkdir(parents=True, exist_ok=True)
        statistics = []
        if len(images) != expected_count:
            raise AssertionError(
                f"{method} produced {len(images)} images, expected {expected_count}"
            )
        for index, image in enumerate(images):
            values = np.asarray(image.convert("RGB"), dtype=np.float32)
            if not np.isfinite(values).all():
                raise FloatingPointError(f"{method} image {index} is non-finite")
            standard_deviation = float(values.std())
            if standard_deviation < 0.1:
                raise FloatingPointError(
                    f"{method} image {index} is effectively constant "
                    f"(std={standard_deviation:.6f})"
                )
            path = method_directory / f"p{index:02d}.png"
            image.save(path)
            statistics.append({
                "path": str(path),
                "size": list(image.size),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "standard_deviation": standard_deviation,
            })
        records[method] = statistics
    return records


def save_comparison_grid(images_by_method, output_directory, profile_name):
    """Save a paper-friendly grid with one column per method."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    methods = list(images_by_method)
    count = len(images_by_method[methods[0]])
    fig, axes = plt.subplots(
        count,
        len(methods),
        figsize=(3.0 * len(methods), 3.0 * count),
        squeeze=False,
        facecolor="white",
    )
    for column, method in enumerate(methods):
        for row, image in enumerate(images_by_method[method]):
            axis = axes[row, column]
            axis.imshow(image)
            axis.axis("off")
    fig.suptitle(profile_name, fontsize=11, y=0.998)
    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.94,
        wspace=0.025,
        hspace=0.025,
    )
    fig.canvas.draw()
    for column, method in enumerate(methods):
        boxes = [axes[row, column].get_position() for row in range(count)]
        x0 = min(box.x0 for box in boxes)
        y0 = min(box.y0 for box in boxes)
        x1 = max(box.x1 for box in boxes)
        y1 = max(box.y1 for box in boxes)
        is_ours = method == "rho_star"
        fig.text(
            (x0 + x1) / 2,
            y1 + 0.006,
            METHOD_LABELS[method],
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="semibold",
            color="#ef4b3f" if is_ours else "black",
        )
        if is_ours:
            padding = 0.003
            fig.add_artist(Rectangle(
                (x0 - padding, y0 - padding),
                (x1 - x0) + 2 * padding,
                (y1 - y0) + 2 * padding,
                transform=fig.transFigure,
                fill=False,
                clip_on=False,
                edgecolor="#ef4b3f",
                linewidth=1.6,
                linestyle=(0, (4, 3)),
            ))
    png = output_directory / "comparison.png"
    pdf = output_directory / "comparison.pdf"
    fig.savefig(png, dpi=250, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png, pdf


def run_generation_comparison(profile, methods, prompt, steps, count):
    import torch
    from evaluation.metrics import PromptMetrics
    from generation import ddim, model
    from methods import cads, rho_star, tpso
    from methods.adaptive_pullback import sample_adaptive_pullback
    from pullback import basis as pullback_basis
    from pullback import directions

    model.load_model(
        profile.base_model,
        family=profile.family,
        local_files_only=True,
    )
    model.set_unet_particle_batch_size(UNET_PARTICLE_BATCH_SIZE)
    positive, negative, real_tokens = model.encode_prompt(
        prompt, NEGATIVE_PROMPT
    )
    initial = model.make_initial_latents(
        count,
        profile.resolution,
        profile.resolution,
        seed=GENERATION_INITIAL_SEED,
    )
    rank = max(PULLBACK_MINIMUM_RANK, count)
    basis = None
    selection = None
    if "rho_star" in methods:
        anchor, basis_timestep, _ = directions.compute_clean_anchor(
            initial[:1],
            positive,
            negative,
            steps,
            GUIDANCE_SCALE,
            ETA,
            NOISE_SEED_BASE,
            PULLBACK_BASIS_TIMESTEP,
        )
        basis, _ = pullback_basis.compute_pullback_basis(
            anchor,
            basis_timestep,
            positive,
            real_tokens,
            rank=rank,
            number_of_iterations=PULLBACK_BASIS_ITERATIONS,
            seed=PULLBACK_BASIS_SEED,
            finite_difference_epsilon=PULLBACK_FINITE_DIFFERENCE_EPSILON,
            progress_label="comparison pullback basis",
        )
        probe_latents, probe_timestep, _ = directions.compute_clean_anchor(
            initial,
            positive,
            negative,
            steps,
            GUIDANCE_SCALE,
            ETA,
            NOISE_SEED_BASE,
            RHO_STAR_PROBE_TIMESTEP,
        )
        candidates = MODEL_SPECIFIC_PARAMETERS[profile.family][
            "rho_star_candidate_rhos"
        ]
        full_directions = rho_star.make_full_scale_directions(
            basis,
            positive,
            real_tokens,
            count,
            PULLBACK_MODE,
            PULLBACK_DIRECTION_SEED,
        )
        probe = rho_star.probe_rho_candidates(
            probe_latents,
            probe_timestep,
            positive,
            negative,
            real_tokens,
            full_directions,
            [0.0, *candidates],
            GUIDANCE_SCALE,
            schedule_start=PULLBACK_START,
            schedule_end=PULLBACK_END,
            schedule_power=PULLBACK_SCHEDULE_POWER,
        )
        metrics = PromptMetrics()
        probe = rho_star.add_clip_dino_probe_features(
            probe,
            prompt,
            metrics,
            decode_batch_size=RHO_STAR_PROBE_DECODE_BATCH_SIZE,
        )
        selection = rho_star.select_rho_combination_clip_dino(
            probe,
            max_clip_drop=GENERATION_RHO_STAR_MAX_CLIP_DROP,
            selectable_rhos=candidates,
            search_strategy=RHO_STAR_SEARCH_STRATEGY,
            constraint_fallback=RHO_STAR_CONSTRAINT_FALLBACK,
        )
        del metrics, probe, probe_latents, full_directions
        gc.collect()
        torch.cuda.empty_cache()

    images = {}
    for method in methods:
        print(f"running {method}", flush=True)
        if method == "clean_ddim":
            latents = ddim.sample_clean_ddim(
                initial,
                positive,
                negative,
                steps,
                GUIDANCE_SCALE,
                ETA,
                NOISE_SEED_BASE,
                progress=False,
            )
        elif method == "cads":
            latents = cads.sample_cads(
                initial,
                positive,
                negative,
                steps,
                GUIDANCE_SCALE,
                ETA,
                NOISE_SEED_BASE,
                start=CADS_START,
                end=CADS_END,
                noise_scale=CADS_NOISE_SCALE,
                psi=CADS_PSI,
                noise_seed=CADS_CONDITION_SEED,
                persistence=CADS_PERSISTENCE,
                use_rescale=CADS_USE_RESCALE,
                progress=False,
            )
        elif method == "tpso":
            optimized = tpso.optimize_token_offsets(
                prompt,
                count,
                kappa=TPSO_KAPPA,
                sigma=TPSO_SIGMA,
                diversity_weight=TPSO_DIVERSITY_WEIGHT,
                learning_rate=TPSO_LEARNING_RATE,
                max_steps=TPSO_MAX_STEPS,
                min_steps=TPSO_MIN_STEPS,
                patience=TPSO_PATIENCE,
                seed=TPSO_SEED,
                log_every=TPSO_LOG_EVERY,
            )
            latents, _ = tpso.sample_tpso(
                initial,
                positive,
                negative,
                optimized,
                steps,
                GUIDANCE_SCALE,
                ETA,
                NOISE_SEED_BASE,
                ratio=TPSO_RATIO,
                progress=False,
            )
            del optimized
        elif method == "rho_star":
            latents, _ = sample_adaptive_pullback(
                initial,
                positive,
                negative,
                real_tokens,
                basis,
                steps,
                GUIDANCE_SCALE,
                ETA,
                NOISE_SEED_BASE,
                particle_rho=selection["selected_rhos"],
                start=PULLBACK_START,
                end=PULLBACK_END,
                schedule_power=PULLBACK_SCHEDULE_POWER,
                mode=PULLBACK_MODE,
                direction_seed=PULLBACK_DIRECTION_SEED,
                number_of_refreshes=PULLBACK_NUMBER_OF_REFRESHES,
                intermediate_rank=rank,
                intermediate_iterations=PULLBACK_INTERMEDIATE_ITERATIONS,
                intermediate_seed=PULLBACK_INTERMEDIATE_SEED,
                transition_steps=PULLBACK_TRANSITION_STEPS,
                finite_difference_epsilon=PULLBACK_FINITE_DIFFERENCE_EPSILON,
                progress=False,
            )
        else:
            raise AssertionError(method)
        images[method] = model.decode_latents(latents)
        del latents
        torch.cuda.empty_cache()
    return images, model.pipe


def run_inpainting_comparison(profile, methods, example_key, steps, count):
    import torch
    from evaluation.inpainting_metrics import InpaintingMetrics
    from inpainting import data, experiment, model
    from inpainting.config import InpaintingConfig

    rhos = MODEL_SPECIFIC_PARAMETERS[profile.family][
        "rho_star_candidate_rhos"
    ]
    rank = max(PULLBACK_MINIMUM_RANK, count)
    config = InpaintingConfig(
        num_particles=count,
        resolution=min(
            profile.resolution, INPAINTING_COMPARISON_RESOLUTION_CAP
        ),
        ddim_steps=steps,
        eta=ETA,
        initial_seed=INPAINTING_INITIAL_SEED,
        noise_seed_base=NOISE_SEED_BASE,
        cads_noise_scale=CADS_NOISE_SCALE,
        cads_start=CADS_START,
        cads_end=CADS_END,
        cads_rescale_factor=CADS_RESCALE_FACTOR,
        cads_condition_seed=CADS_CONDITION_SEED,
        pullback_rank=rank,
        pullback_basis_timestep=PULLBACK_BASIS_TIMESTEP,
        pullback_basis_seed=PULLBACK_BASIS_SEED,
        pullback_basis_iterations=PULLBACK_BASIS_ITERATIONS,
        pullback_rho=rhos[0],
        pullback_start=PULLBACK_START,
        pullback_end=PULLBACK_END,
        pullback_schedule_power=PULLBACK_SCHEDULE_POWER,
        pullback_direction_seed=PULLBACK_DIRECTION_SEED,
        pullback_refreshes=PULLBACK_NUMBER_OF_REFRESHES,
        pullback_intermediate_rank=rank,
        pullback_intermediate_iterations=PULLBACK_INTERMEDIATE_ITERATIONS,
        pullback_intermediate_seed=PULLBACK_INTERMEDIATE_SEED,
        pullback_transition_steps=PULLBACK_TRANSITION_STEPS,
        pullback_anchor_particle=PULLBACK_ANCHOR_PARTICLE,
        pullback_response_region=PULLBACK_RESPONSE_REGION,
        rho_star_probe_timestep=RHO_STAR_PROBE_TIMESTEP,
        rho_star_candidate_rhos=rhos,
        rho_star_max_clip_drop=INPAINTING_RHO_STAR_MAX_CLIP_DROP,
        rho_star_search_strategy=RHO_STAR_SEARCH_STRATEGY,
        rho_star_max_combinations=RHO_STAR_MAX_COMBINATIONS,
        rho_star_beam_width=RHO_STAR_BEAM_WIDTH,
        rho_star_constraint_fallback=RHO_STAR_CONSTRAINT_FALLBACK,
        tpso_kappa=TPSO_KAPPA,
        tpso_sigma=TPSO_SIGMA,
        tpso_diversity_weight=TPSO_DIVERSITY_WEIGHT,
        tpso_learning_rate=TPSO_LEARNING_RATE,
        tpso_max_steps=TPSO_MAX_STEPS,
        tpso_min_steps=TPSO_MIN_STEPS,
        tpso_patience=TPSO_PATIENCE,
        tpso_initial_std=TPSO_INITIALIZATION_STD,
        tpso_seed=TPSO_SEED,
        tpso_ratio=TPSO_RATIO,
    )
    config.validate()
    pipe = model.load_pipeline()
    example = data.load_example(example_key)
    sample = experiment.prepare_example(pipe, example, config)
    metrics = None
    basis = None
    if "rho_star" in methods:
        metrics = InpaintingMetrics(device=str(pipe._execution_device))
        basis, _ = experiment.compute_initial_basis(pipe, sample, config)

    images = {}
    optimized = None
    for method in methods:
        print(f"running {method}", flush=True)
        if method == "clean_ddim":
            _, blended = experiment.run_clean_ddim(
                pipe, sample, config, progress=False
            )
        elif method == "cads":
            _, blended = experiment.run_cads(
                pipe, sample, config, progress=False
            )
        elif method == "tpso":
            if optimized is None:
                optimized = experiment.optimize_tpso(pipe, sample, config)
            _, blended, _ = experiment.run_tpso(
                pipe, sample, optimized, config, progress=False
            )
            optimized = None
        elif method == "rho_star":
            _, blended, _ = experiment.run_rho_star_pullback(
                pipe, sample, basis, config, metrics, progress=False
            )
        else:
            raise AssertionError(method)
        images[method] = blended
        torch.cuda.empty_cache()
    return images, pipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    parser.add_argument("--methods", default="all", type=parse_methods)
    parser.add_argument("--base-model")
    parser.add_argument("--brushnet-model")
    parser.add_argument("--brushbench-root")
    parser.add_argument("--example", default=DEFAULT_EXAMPLE_KEY)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_DDIM_STEPS)
    parser.add_argument(
        "--num-particles", type=int, default=DEFAULT_NUM_PARTICLES
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.steps < MINIMUM_DDIM_STEPS:
        parser.error(
            f"--steps must be at least {MINIMUM_DDIM_STEPS} so scheduled "
            "methods are exercised"
        )
    if args.num_particles < MINIMUM_NUM_PARTICLES:
        parser.error(
            f"--num-particles must be at least {MINIMUM_NUM_PARTICLES}"
        )

    profile = get_profile(
        args.profile,
        base_model=args.base_model,
        brushnet_model=args.brushnet_model,
        brushbench_root=args.brushbench_root,
    )
    configure_inpainting_environment(profile)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the comparison")
    output_directory = args.output_dir or (
        REPOSITORY_ROOT / "outputs" / "comparisons" / profile.name
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pipe = None
    try:
        if profile.task == "generation":
            images, pipe = run_generation_comparison(
                profile,
                args.methods,
                args.prompt,
                args.steps,
                args.num_particles,
            )
        else:
            images, pipe = run_inpainting_comparison(
                profile,
                args.methods,
                args.example,
                args.steps,
                args.num_particles,
            )
        records = validate_and_save(
            images, output_directory, args.num_particles
        )
        comparison_png, comparison_pdf = save_comparison_grid(
            images, output_directory, profile.name
        )
        result = {
            "profile": profile.name,
            "methods": args.methods,
            "steps": args.steps,
            "num_particles": args.num_particles,
            "seconds": time.perf_counter() - started,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "images": records,
            "comparison_png": str(comparison_png),
            "comparison_pdf": str(comparison_pdf),
            "status": "passed",
        }
        manifest = output_directory / "result.json"
        manifest.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
