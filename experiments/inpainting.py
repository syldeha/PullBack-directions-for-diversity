"""Resumable long evaluation of DDIM, CADS, pullback, and TPSO on BrushBench.

Edit ``configs/inpainting.py`` and run this module. Each (example, method) result is
checkpointed immediately, so rerunning the same RUN_NAME resumes incomplete
work and rebuilds the cumulative aggregate tables.
"""

import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pullback-matplotlib-cache")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(
    os.environ.get(
        "PULLBACK_INPAINTING_OUTPUT",
        REPOSITORY_ROOT / "outputs" / "inpainting",
    )
)

from evaluation.inpainting_metrics import InpaintingMetrics
from inpainting import data, experiment, model, tpso


METHODS = (
    "clean_ddim",
    "cads",
    "adaptive_pullback",
    "rho_star_pullback",
    "tpso",
)
SCALAR_METRICS = (
    "lpips_mean",
    "lpips_min",
    "dino_sim_mean",
    "dino_sim_max",
    "vendi",
    "clip",
    "aesthetic",
    "mss",
    "mask_dino_sim_mean",
    "mask_dino_sim_max",
    "mask_vendi",
    "mask_clip",
    "mask_mss",
    "raw_outside_mae",
    "raw_outside_psnr",
    "seconds",
)


def load_config(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("inpainting_long_eval_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import configuration: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
    temporary.replace(path)


def resolve_example_keys(config_module):
    available = sorted(data.available_keys())
    requested = [str(key) for key in config_module.EXAMPLE_KEYS]
    if requested:
        requested = [f"{int(key):09d}" if key.isdigit() else key for key in requested]
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"BrushBench images are not available locally: {missing}")
        keys = requested
    else:
        keys = available

    seed = getattr(config_module, "EXAMPLE_ORDER_SEED", None)
    if seed is not None:
        random.Random(int(seed)).shuffle(keys)
    maximum = config_module.MAX_EXAMPLES
    if maximum is not None:
        if int(maximum) < 1:
            raise ValueError("MAX_EXAMPLES must be positive or None")
        keys = keys[:int(maximum)]
    if not keys:
        raise ValueError("No locally available BrushBench examples were selected")
    return keys


def validate_config(config_module, keys):
    config_module.EXPERIMENT.validate()
    selected = list(config_module.SELECTED_METHODS)
    if not selected:
        raise ValueError("SELECTED_METHODS cannot be empty")
    if len(selected) != len(set(selected)):
        raise ValueError("SELECTED_METHODS contains duplicates")
    unknown = sorted(set(selected) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; choose from {METHODS}")
    if {"adaptive_pullback", "rho_star_pullback"} & set(selected):
        n = config_module.EXPERIMENT.num_particles
        if config_module.EXPERIMENT.pullback_rank < n:
            raise ValueError("pullback_rank must be at least num_particles")
        if config_module.EXPERIMENT.pullback_intermediate_rank < n:
            raise ValueError("pullback_intermediate_rank must be at least num_particles")
    if not keys:
        raise ValueError("No examples selected")


def experiment_identity(config_module, keys):
    return {
        "format_version": 1,
        "run_name": config_module.RUN_NAME,
        "example_keys": keys,
        "selected_methods": list(config_module.SELECTED_METHODS),
        "experiment": asdict(config_module.EXPERIMENT),
        "compute_mask_crop_metrics": bool(config_module.COMPUTE_MASK_CROP_METRICS),
        "compute_raw_outside_preservation": bool(
            config_module.COMPUTE_RAW_OUTSIDE_PRESERVATION
        ),
    }


def prepare_output(config_module, identity):
    output_root = OUTPUT_ROOT / config_module.RUN_NAME
    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "config.json"
    if config_path.exists():
        with config_path.open() as handle:
            previous = json.load(handle)
        if previous != json_safe(identity):
            raise RuntimeError(
                f"Configuration changed for {output_root}. Change RUN_NAME to "
                "start a new experiment, or restore the previous configuration."
            )
    else:
        write_json(config_path, identity)
    return output_root


def method_directory(output_root, example_key, method):
    return output_root / "examples" / example_key / method


def load_completed(output_root, example_key, method, num_particles):
    directory = method_directory(output_root, example_key, method)
    result_path = directory / "result.json"
    blended_paths = [directory / "blended" / f"p{i:02d}.png" for i in range(num_particles)]
    if not result_path.exists() or not all(path.exists() for path in blended_paths):
        return None
    with result_path.open() as handle:
        result = json.load(handle)
    images = [Image.open(path).convert("RGB") for path in blended_paths]
    return result, images


def save_example_context(output_root, example, prep):
    directory = output_root / "examples" / example.key
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "source.png"
    mask_path = directory / "mask.png"
    if not source_path.exists():
        prep.source_image.save(source_path)
    if not mask_path.exists():
        prep.edit_mask_image.save(mask_path)


def outside_preservation(raw_images, prep):
    source = np.asarray(prep.source_image.convert("RGB"), dtype=np.float32) / 255.0
    mask = np.asarray(prep.edit_mask_image) > 127
    outside = ~mask
    if not outside.any():
        return {"raw_outside_mae": float("nan"), "raw_outside_psnr": float("nan")}

    maes = []
    psnrs = []
    for image in raw_images:
        generated = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        difference = generated[outside] - source[outside]
        mae = float(np.abs(difference).mean())
        mse = float(np.square(difference).mean())
        psnr = float("inf") if mse == 0 else float(-10.0 * math.log10(mse))
        maes.append(mae)
        psnrs.append(psnr)
    return {
        "raw_outside_mae": float(np.mean(maes)),
        "raw_outside_psnr": float(np.mean(psnrs)),
    }


def evaluate_images(metrics, blended, raw, prep, config_module):
    values = metrics.full_metrics(blended, prep.caption, prep.edit_mask_image)

    if config_module.COMPUTE_MASK_CROP_METRICS:
        crops = [
            metrics.mask_bbox_crop(image, prep.edit_mask_image)
            for image in blended
        ]
        clip_crop = metrics.clip_metrics(crops, prep.caption)
        dino_crop = metrics.dino_metrics(crops)
        values.update({
            "mask_dino_sim_mean": dino_crop["dino_sim_mean"],
            "mask_dino_sim_max": dino_crop["dino_sim_max"],
            "mask_vendi": clip_crop["vendi"],
            "mask_clip": float(np.mean(clip_crop["clip_all"])),
            "mask_mss": clip_crop["mss"],
        })

    if config_module.COMPUTE_RAW_OUTSIDE_PRESERVATION:
        values.update(outside_preservation(raw, prep))
    return values


def run_method(method, pipe, prep, config, cache_dir, metrics, progress):
    details = {}
    history = None
    started = time.perf_counter()

    if method == "clean_ddim":
        raw, blended = experiment.run_clean_ddim(
            pipe, prep, config, progress=progress
        )

    elif method == "cads":
        raw, blended = experiment.run_cads(
            pipe, prep, config, progress=progress
        )

    elif method == "adaptive_pullback":
        basis_started = time.perf_counter()
        basis, eigenvalues = experiment.compute_initial_basis(
            pipe, prep, config, cache_dir=cache_dir
        )
        basis_seconds = time.perf_counter() - basis_started
        sample_started = time.perf_counter()
        raw, blended = experiment.run_adaptive_pullback(
            pipe, prep, basis, config, progress=progress
        )
        details = {
            "basis_seconds": basis_seconds,
            "sampling_seconds": time.perf_counter() - sample_started,
            "pullback_eigenvalues": eigenvalues,
        }

    elif method == "rho_star_pullback":
        basis_started = time.perf_counter()
        basis, eigenvalues = experiment.compute_initial_basis(
            pipe, prep, config, cache_dir=cache_dir
        )
        basis_seconds = time.perf_counter() - basis_started
        raw, blended, details = experiment.run_rho_star_pullback(
            pipe,
            prep,
            basis,
            config,
            metrics,
            progress=progress,
        )
        details.update({
            "basis_seconds": basis_seconds,
            "pullback_eigenvalues": eigenvalues,
        })

    elif method == "tpso":
        verification = tpso.verify_clean_encoding(pipe, prep)
        optimization_started = time.perf_counter()
        optimized = experiment.optimize_tpso(pipe, prep, config)
        optimization_seconds = time.perf_counter() - optimization_started
        sample_started = time.perf_counter()
        raw, blended, alpha_trace = experiment.run_tpso(
            pipe, prep, optimized, config, progress=progress
        )
        history = optimized.history
        details = {
            "zero_offset_verification": verification,
            "optimization_seconds": optimization_seconds,
            "sampling_seconds": time.perf_counter() - sample_started,
            "steps_run": optimized.steps_run,
            "offset_norms": optimized.offset_norms,
            "final_objective": optimized.history[-1],
            "alpha_trace": alpha_trace,
        }
    else:
        raise ValueError(method)

    return raw, blended, details, history, time.perf_counter() - started


def save_method_result(
    output_root,
    example,
    method,
    raw,
    blended,
    metrics,
    details,
    history,
    seconds,
    save_raw,
):
    directory = method_directory(output_root, example.key, method)
    blended_dir = directory / "blended"
    blended_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(blended):
        image.save(blended_dir / f"p{index:02d}.png")

    if save_raw:
        raw_dir = directory / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(raw):
            image.save(raw_dir / f"p{index:02d}.png")

    if history is not None:
        pd.DataFrame(history).to_csv(directory / "tpso_history.csv", index=False)

    scalar_row = {
        "example_key": example.key,
        "prompt": example.prompt,
        "mask_fraction": example.mask_fraction,
        "method": method,
        "seconds": float(seconds),
    }
    for name in SCALAR_METRICS:
        if name == "seconds":
            continue
        value = metrics.get(name, float("nan"))
        scalar_row[name] = float(value) if value is not None else float("nan")

    result = {
        **scalar_row,
        "metrics": metrics,
        "details": details,
        "completed_at": time.time(),
    }
    write_json(directory / "result.json", result)
    return result


def scalar_row(result):
    row = {
        "example_key": result["example_key"],
        "prompt": result["prompt"],
        "mask_fraction": result["mask_fraction"],
        "method": result["method"],
    }
    for name in SCALAR_METRICS:
        value = result.get(name)
        row[name] = float("nan") if value is None else float(value)
    return row


def collect_completed(output_root, keys, methods, num_particles):
    rows = []
    for key in keys:
        for method in methods:
            completed = load_completed(output_root, key, method, num_particles)
            if completed is not None:
                result, _ = completed
                rows.append(scalar_row(result))
    return rows


def rebuild_tables(output_root, rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, pd.DataFrame()
    frame = frame.sort_values(["example_key", "method"]).reset_index(drop=True)
    frame.to_csv(output_root / "per_example_metrics.csv", index=False)

    summary_rows = []
    for method, group in frame.groupby("method", sort=False):
        row = {"method": method, "n_examples": int(len(group))}
        for metric in SCALAR_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else float("nan")
            row[f"{metric}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
            row[f"{metric}_min"] = float(finite.min()) if len(finite) else float("nan")
            row[f"{metric}_max"] = float(finite.max()) if len(finite) else float("nan")
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "aggregate_metrics.csv", index=False)
    write_json(output_root / "aggregate_metrics.json", summary.to_dict(orient="records"))
    return frame, summary


def print_current_result(result):
    print(
        f"{result['method']}: {result['seconds'] / 60.0:.2f} min "
        f"CLIP={result.get('clip', float('nan')):.3f} "
        f"maskCLIP={result.get('mask_clip', float('nan')):.3f} "
        f"DINO={result.get('dino_sim_mean', float('nan')):.4f} "
        f"maskDINO={result.get('mask_dino_sim_mean', float('nan')):.4f} "
        f"LPIPS={result.get('lpips_mean', float('nan')):.4f} "
        f"Vendi={result.get('vendi', float('nan')):.3f}",
        flush=True,
    )


def print_aggregate(summary):
    if summary.empty:
        return
    columns = [
        "method",
        "n_examples",
        "clip_mean",
        "mask_clip_mean",
        "dino_sim_mean_mean",
        "mask_dino_sim_mean_mean",
        "lpips_mean_mean",
        "mss_mean",
        "mask_mss_mean",
        "vendi_mean",
        "seconds_mean",
    ]
    columns = [column for column in columns if column in summary]
    print("\nCumulative aggregate", flush=True)
    print(summary[columns].round(4).to_string(index=False), flush=True)


def build_grid(output_root, example, methods, num_particles):
    images = {}
    for method in methods:
        completed = load_completed(output_root, example.key, method, num_particles)
        if completed is None:
            return None
        _, images[method] = completed
    figure, axes = plt.subplots(
        len(methods),
        num_particles,
        figsize=(3.4 * num_particles, 3.5 * len(methods)),
        squeeze=False,
    )
    for row, method in enumerate(methods):
        for column in range(num_particles):
            axis = axes[row, column]
            axis.axis("off")
            axis.imshow(images[method][column])
            axis.set_title(f"{method} p{column}", fontsize=9)
    figure.suptitle(f"{example.key}: {example.prompt}", fontsize=12)
    plt.tight_layout()
    grid_path = output_root / "examples" / example.key / "comparison_grid.png"
    figure.savefig(grid_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return grid_path


def init_wandb(config_module, identity):
    if not config_module.USE_WANDB:
        return None
    try:
        import wandb
    except ImportError:
        print("WARNING: wandb is unavailable; continuing without it", flush=True)
        return None
    try:
        return wandb.init(
            project=config_module.WANDB_PROJECT,
            entity=config_module.WANDB_ENTITY,
            name=config_module.RUN_NAME,
            id=config_module.RUN_NAME,
            resume="allow",
            mode=config_module.WANDB_MODE,
            config=json_safe(identity),
        )
    except Exception as error:
        print(
            f"WARNING: W&B initialization failed ({type(error).__name__}: {error}); "
            "continuing with local checkpoints",
            flush=True,
        )
        return None


def log_example_wandb(run, example_index, example, results, grid_path, config_module):
    if run is None:
        return
    payload = {
        "example_index": example_index,
        "example/key": example.key,
        "example/mask_fraction": example.mask_fraction,
    }
    for method, result in results.items():
        for metric in SCALAR_METRICS:
            value = result.get(metric)
            if value is not None and math.isfinite(float(value)):
                payload[f"metrics/{metric}/{method}"] = float(value)
    if (
        config_module.WANDB_LOG_GRIDS
        and grid_path is not None
        and (example_index == 0 or (example_index + 1) % config_module.WANDB_GRID_EVERY == 0)
    ):
        import wandb
        payload["examples/comparison_grid"] = wandb.Image(
            str(grid_path), caption=f"{example.key}: {example.prompt}"
        )
    try:
        run.log(payload, step=example_index)
    except Exception as error:
        print(
            f"WARNING: W&B logging failed ({type(error).__name__}: {error})",
            flush=True,
        )


def update_wandb_summary(run, summary):
    if run is None or summary.empty:
        return
    try:
        for _, row in summary.iterrows():
            method = row["method"]
            for column, value in row.items():
                if column == "method" or value is None:
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
                    run.summary[f"aggregate/{method}/{column}"] = float(value)
    except Exception as error:
        print(
            f"WARNING: W&B summary update failed ({type(error).__name__}: {error})",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=REPOSITORY_ROOT / "configs" / "inpainting.py",
        type=Path,
        help="Path to the editable Python configuration file",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and print the run without loading model weights",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild CSV summaries from completed checkpoints",
    )
    args = parser.parse_args()

    config_module = load_config(args.config)
    keys = resolve_example_keys(config_module)
    validate_config(config_module, keys)
    identity = experiment_identity(config_module, keys)
    methods = list(config_module.SELECTED_METHODS)
    n = config_module.EXPERIMENT.num_particles

    if args.check:
        output_root = OUTPUT_ROOT / config_module.RUN_NAME
        print(
            f"run={config_module.RUN_NAME} examples={len(keys)} methods={methods} "
            f"particles={n}",
            flush=True,
        )
        print("output:", output_root, flush=True)
        print("first keys:", keys[:10])
        print("configuration is valid; no run directory was created")
        return

    output_root = prepare_output(config_module, identity)

    completed_rows = collect_completed(output_root, keys, methods, n)
    print(
        f"run={config_module.RUN_NAME} examples={len(keys)} methods={methods} "
        f"particles={n} completed={len(completed_rows)}/{len(keys) * len(methods)}",
        flush=True,
    )
    print("output:", output_root, flush=True)

    frame, summary = rebuild_tables(output_root, completed_rows)
    if args.aggregate_only:
        print_aggregate(summary)
        return

    pending = len(keys) * len(methods) - len(completed_rows)
    if pending == 0:
        print_aggregate(summary)
        print("all selected results are already complete")
        return

    pipe = model.load_pipeline()
    metrics = InpaintingMetrics(device=str(pipe._execution_device))
    wandb_run = init_wandb(config_module, identity)
    cache_dir = output_root / "basis_cache"

    try:
        for example_index, key in enumerate(keys):
            example = data.load_example(key)
            print(
                f"\n===== BrushBench {example_index + 1}/{len(keys)} "
                f"key={example.key} mask={100 * example.mask_fraction:.1f}% =====",
                flush=True,
            )
            print(example.prompt, flush=True)
            prep = experiment.prepare_example(
                pipe, example, config_module.EXPERIMENT
            )
            save_example_context(output_root, example, prep)
            prompt_results = {}

            for method in methods:
                completed = load_completed(output_root, key, method, n)
                if completed is not None:
                    result, _ = completed
                    prompt_results[method] = result
                    print("skipping completed:", method, flush=True)
                    continue

                print("running:", method, flush=True)
                try:
                    raw, blended, details, history, seconds = run_method(
                        method,
                        pipe,
                        prep,
                        config_module.EXPERIMENT,
                        cache_dir,
                        metrics,
                        config_module.SHOW_SAMPLER_PROGRESS,
                    )
                    metric_values = evaluate_images(
                        metrics, blended, raw, prep, config_module
                    )
                    result = save_method_result(
                        output_root,
                        example,
                        method,
                        raw,
                        blended,
                        metric_values,
                        details,
                        history,
                        seconds,
                        config_module.SAVE_RAW_IMAGES,
                    )
                except Exception:
                    failure = {
                        "example_key": key,
                        "method": method,
                        "traceback": traceback.format_exc(),
                        "failed_at": time.time(),
                    }
                    write_json(
                        method_directory(output_root, key, method) / "failure.json",
                        failure,
                    )
                    raise

                prompt_results[method] = result
                print_current_result(result)
                completed_rows = collect_completed(output_root, keys, methods, n)
                frame, summary = rebuild_tables(output_root, completed_rows)
                print_aggregate(summary)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            grid_path = None
            if config_module.SAVE_COMPARISON_GRIDS:
                grid_path = build_grid(output_root, example, methods, n)
            log_example_wandb(
                wandb_run,
                example_index,
                example,
                prompt_results,
                grid_path,
                config_module,
            )
            update_wandb_summary(wandb_run, summary)

    finally:
        completed_rows = collect_completed(output_root, keys, methods, n)
        frame, summary = rebuild_tables(output_root, completed_rows)
        print_aggregate(summary)
        update_wandb_summary(wandb_run, summary)
        if wandb_run is not None:
            try:
                import wandb
                table = wandb.Table(dataframe=frame)
                wandb_run.log({"results/per_example": table})
                wandb_run.finish()
            except Exception as error:
                print(
                    f"WARNING: final W&B upload failed ({type(error).__name__}: {error})",
                    flush=True,
                )

    print("\nevaluation complete:", output_root, flush=True)


if __name__ == "__main__":
    main()
