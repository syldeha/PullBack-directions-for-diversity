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
    model.set_unet_particle_batch_size(1)
    positive, negative, real_tokens = model.encode_prompt(prompt, "")
    initial = model.make_initial_latents(
        count, profile.resolution, profile.resolution, seed=12345
    )
    rank = max(2, count)
    basis = None
    selection = None
    if "rho_star" in methods:
        anchor, basis_timestep, _ = directions.compute_clean_anchor(
            initial[:1], positive, negative, steps, 7.5, 0.0, 20800, 600
        )
        basis, _ = pullback_basis.compute_pullback_basis(
            anchor,
            basis_timestep,
            positive,
            real_tokens,
            rank=rank,
            number_of_iterations=1,
            seed=515,
            finite_difference_epsilon=0.5,
            progress_label="comparison pullback basis",
        )
        probe_latents, probe_timestep, _ = directions.compute_clean_anchor(
            initial, positive, negative, steps, 7.5, 0.0, 20800, 699
        )
        candidates = [0.10, 0.20] if profile.family == "sd15" else [1.0, 2.0]
        full_directions = rho_star.make_full_scale_directions(
            basis, positive, real_tokens, count, "disjoint", 777
        )
        probe = rho_star.probe_rho_candidates(
            probe_latents,
            probe_timestep,
            positive,
            negative,
            real_tokens,
            full_directions,
            [0.0] + candidates,
            7.5,
            schedule_start=999,
            schedule_end=500,
            schedule_power=2.0,
        )
        metrics = PromptMetrics()
        probe = rho_star.add_clip_dino_probe_features(
            probe, prompt, metrics, decode_batch_size=2
        )
        selection = rho_star.select_rho_combination_clip_dino(
            probe,
            max_clip_drop=0.5,
            selectable_rhos=candidates,
            search_strategy="exact",
            constraint_fallback="minimum_selectable",
        )
        del metrics, probe, probe_latents, full_directions
        gc.collect()
        torch.cuda.empty_cache()

    images = {}
    for method in methods:
        print(f"running {method}", flush=True)
        if method == "clean_ddim":
            latents = ddim.sample_clean_ddim(
                initial, positive, negative, steps, 7.5, 0.0, 20800, progress=False
            )
        elif method == "cads":
            latents = cads.sample_cads(
                initial,
                positive,
                negative,
                steps,
                7.5,
                0.0,
                20800,
                start=900,
                end=600,
                noise_scale=0.15,
                psi=1.0,
                noise_seed=999,
                persistence="fresh",
                use_rescale=True,
                progress=False,
            )
        elif method == "tpso":
            optimized = tpso.optimize_token_offsets(
                prompt,
                count,
                kappa=0.80,
                sigma=0.01,
                diversity_weight=1.0,
                learning_rate=1e-3,
                max_steps=4,
                min_steps=2,
                patience=2,
                seed=3407,
                log_every=1,
            )
            latents, _ = tpso.sample_tpso(
                initial,
                positive,
                negative,
                optimized,
                steps,
                7.5,
                0.0,
                20800,
                ratio=0.4,
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
                7.5,
                0.0,
                20800,
                particle_rho=selection["selected_rhos"],
                start=999,
                end=500,
                schedule_power=2.0,
                mode="disjoint",
                direction_seed=777,
                number_of_refreshes=1,
                intermediate_rank=rank,
                intermediate_iterations=1,
                intermediate_seed=1515,
                transition_steps=1,
                finite_difference_epsilon=0.5,
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

    rhos = (0.10, 0.20) if profile.family == "sd15" else (1.0, 2.0)
    config = InpaintingConfig(
        num_particles=count,
        resolution=min(profile.resolution, 512),
        ddim_steps=steps,
        pullback_rank=max(2, count),
        pullback_basis_timestep=600,
        pullback_basis_iterations=1,
        pullback_rho=rhos[0],
        pullback_refreshes=1,
        pullback_intermediate_rank=max(2, count),
        pullback_intermediate_iterations=1,
        pullback_transition_steps=1,
        rho_star_candidate_rhos=rhos,
        rho_star_search_strategy="exact",
        tpso_max_steps=4,
        tpso_min_steps=2,
        tpso_patience=2,
    )
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
    parser.add_argument("--example", default="000000089")
    parser.add_argument(
        "--prompt",
        default="A red fox standing in an autumn forest, photorealistic",
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--num-particles", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.steps < 10:
        parser.error("--steps must be at least 10 so scheduled methods are exercised")
    if args.num_particles < 2:
        parser.error("--num-particles must be at least 2")

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
