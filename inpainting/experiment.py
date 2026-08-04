"""Readable method entry points for one shared BrushNet experiment state."""

from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from inpainting import cads, model, pullback, tpso


def prepare_example(pipe, example, config):
    """Prepare independent initial particles shared unchanged by all methods."""
    config.validate()
    return model.prepare_sample(
        pipe,
        example.source_image,
        example.mask,
        example.prompt,
        num_particles=config.num_particles,
        resolution=config.resolution,
        ddim_steps=config.ddim_steps,
        start_timestep=999,
        basis_timestep=config.pullback_basis_timestep,
        initial_noise="independent",
        initial_seed=config.initial_seed,
    )


def basis_cache_identity(sample, config, response_region):
    source_digest = hashlib.md5(
        np.asarray(sample.source_image).tobytes()
    ).hexdigest()
    mask_digest = hashlib.md5(
        np.asarray(sample.edit_mask_image).tobytes()
    ).hexdigest()
    return {
        "version": 3,
        "response_region": response_region,
        "caption": sample.caption,
        "source_digest": source_digest,
        "mask_digest": mask_digest,
        "latent_shape": list(sample.initial_latents.shape[1:]),
        "resolution": int(config.resolution),
        "ddim_steps": int(config.ddim_steps),
        "eta": float(config.eta),
        "rank": int(config.pullback_rank),
        "basis_timestep": int(config.pullback_basis_timestep),
        "basis_iterations": int(config.pullback_basis_iterations),
        "basis_seed": int(config.pullback_basis_seed),
        "finite_difference_epsilon": float(pullback.PULLBACK_FD_EPSILON),
        "pullback_chunk": int(pullback.PULLBACK_CHUNK),
        "guidance_scale": float(model.GUIDANCE_SCALE),
        "brushnet_scale": float(model.BRUSHNET_SCALE),
        "negative_prompt": str(model.NEGATIVE_PROMPT),
        "base_model": str(model.BASE_MODEL),
        "brushnet_model": str(model.BRUSHNET_MODEL),
    }


def compute_initial_basis(pipe, sample, config, cache_dir=None):
    """Compute or load the configured global or edit-mask pullback basis."""
    response_region = config.pullback_response_region
    cache_path = None
    identity = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        identity = basis_cache_identity(sample, config, response_region)
        identity_text = json.dumps(identity, sort_keys=True)
        tag = hashlib.md5(identity_text.encode("utf-8")).hexdigest()[:12]
        cache_path = cache_dir / (
            f"basis_v3_{tag}_r{config.pullback_rank}"
            f"_res{config.resolution}_t{config.pullback_basis_timestep}.pt"
        )
        if cache_path.exists():
            saved = torch.load(cache_path, map_location="cpu")
            if (
                saved.get("identity") == identity
                and saved["basis"].shape[1] == sample.real_token_count
            ):
                return (
                    saved["basis"].to(pipe._execution_device),
                    saved["evals"],
                )

    anchor = pullback.compute_anchor(
        pipe,
        sample,
        eta=config.eta,
        noise_seed_base=config.noise_seed_base,
    )
    center = sample.prompt_embed[:, :sample.real_token_count, :].clone()
    basis, eigenvalues = pullback.compute_basis(
        pipe,
        anchor,
        center,
        sample.timesteps[sample.basis_index],
        sample,
        rank=config.pullback_rank,
        seed=config.pullback_basis_seed,
        iterations=config.pullback_basis_iterations,
        response_region=response_region,
    )
    if cache_path is not None:
        torch.save(
            {
                "basis": basis.cpu(),
                "evals": eigenvalues,
                "identity": identity,
            },
            cache_path,
        )
    return basis, eigenvalues


def run_clean_ddim(pipe, sample, config, progress=True):
    clean = sample.prompt_embed.repeat(config.num_particles, 1, 1)
    return pullback.run_scheduled(
        pipe,
        sample,
        clean,
        schedule=None,
        eta=config.eta,
        noise_seed_base=config.noise_seed_base,
        num_particles=config.num_particles,
        progress=progress,
    )


def run_cads(pipe, sample, config, progress=True):
    """Run the original CADS setting used by the long comparison."""
    return cads.run_cads(
        pipe,
        sample,
        noise_scale=config.cads_noise_scale,
        num_particles=config.num_particles,
        eta=config.eta,
        noise_seed_base=config.noise_seed_base,
        start=config.cads_start,
        end=config.cads_end,
        mode="isotropic",
        persistence="fresh",
        negative_mode="isotropic",
        condition_seed=config.cads_condition_seed,
        rescale_factor=config.cads_rescale_factor,
        rescale=True,
        progress=progress,
    )


def run_adaptive_pullback(pipe, sample, basis, config, progress=True):
    """Run fixed-noise adaptive disjoint pullback without rho-star selection."""
    return pullback.run_adaptive(
        pipe,
        sample,
        basis,
        mode="disjoint",
        rho=config.pullback_rho,
        num_particles=config.num_particles,
        eta=config.eta,
        noise_seed_base=config.noise_seed_base,
        schedule=(config.pullback_start, config.pullback_end),
        direction_seed=config.pullback_direction_seed,
        schedule_power=config.pullback_schedule_power,
        num_refreshes=config.pullback_refreshes,
        intermediate_rank=config.pullback_intermediate_rank,
        intermediate_iterations=config.pullback_intermediate_iterations,
        intermediate_seed=config.pullback_intermediate_seed,
        transition_steps=config.pullback_transition_steps,
        anchor_particle=config.pullback_anchor_particle,
        response_region=config.pullback_response_region,
        progress=progress,
    )


def optimize_tpso(pipe, sample, config):
    return tpso.optimize(
        pipe,
        sample.caption,
        num_particles=config.num_particles,
        kappa=config.tpso_kappa,
        sigma=config.tpso_sigma,
        diversity_weight=config.tpso_diversity_weight,
        learning_rate=config.tpso_learning_rate,
        max_steps=config.tpso_max_steps,
        min_steps=config.tpso_min_steps,
        patience=config.tpso_patience,
        initial_std=config.tpso_initial_std,
        seed=config.tpso_seed,
    )


def run_tpso(pipe, sample, optimized, config, progress=True):
    return tpso.run(
        pipe,
        sample,
        optimized,
        num_particles=config.num_particles,
        eta=config.eta,
        noise_seed_base=config.noise_seed_base,
        ratio=config.tpso_ratio,
        progress=progress,
    )
