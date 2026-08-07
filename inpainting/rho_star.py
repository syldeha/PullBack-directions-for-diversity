"""Mask-aware particle-specific pullback scale selection for BrushNet."""

import time

import torch

from inpainting import model, pullback
from methods.rho_star import select_rho_combination_clip_dino


def clean_prefix(pipe, sample, config):
    """Run every original particle with the clean prompt to the probe step."""
    probe_index = model.closest_timestep_index(
        sample.timesteps,
        config.rho_star_probe_timestep,
    )
    if probe_index <= sample.start_index:
        raise ValueError("rho-star probe must occur after the sampling start")
    if probe_index >= len(sample.timesteps) - 1:
        raise ValueError("rho-star probe must occur before the final DDIM step")

    count = config.num_particles
    latents = sample.initial_latents.clone()
    clean = sample.prompt_embed.repeat(count, 1, 1)
    for step in range(sample.start_index, probe_index):
        latents = model.ddim_step(
            pipe,
            latents,
            sample.timesteps[step],
            sample.timesteps[step + 1],
            sample,
            config.eta,
            config.noise_seed_base + step,
            clean,
        )
    return latents, probe_index


def full_scale_directions(pipe, sample, basis, config):
    """Reproduce the fixed disjoint directions used by adaptive sampling."""
    fixed_noise = pullback.make_fixed_prompt_noise(
        pipe,
        sample,
        basis,
        "disjoint",
        config.pullback_direction_seed,
        config.num_particles,
    )
    directions = pullback.project_fixed_prompt_noise(
        basis,
        fixed_noise,
        "disjoint",
    )
    token_norms = sample.prompt_embed[
        0,
        :sample.real_token_count,
        :,
    ].float().norm(dim=-1)
    return directions * token_norms.view(1, -1, 1)


@torch.no_grad()
def probe_candidates(pipe, sample, basis, config, metrics):
    """Evaluate rho candidates with one predicted-clean estimate per candidate."""
    prefix_started = time.perf_counter()
    latents, probe_index = clean_prefix(pipe, sample, config)
    prefix_seconds = time.perf_counter() - prefix_started

    candidate_rhos = [0.0] + [
        float(value) for value in config.rho_star_candidate_rhos
    ]
    count = config.num_particles
    number_of_candidates = len(candidate_rhos)
    directions = full_scale_directions(pipe, sample, basis, config)
    timestep = sample.timesteps[probe_index]
    envelope = pullback.schedule_envelope(
        float(timestep.item()),
        config.pullback_start,
        config.pullback_end,
        power=config.pullback_schedule_power,
    )

    clean = sample.prompt_embed.repeat(count, 1, 1).float()
    conditions = clean[:, None].repeat(1, number_of_candidates, 1, 1)
    rho_tensor = torch.tensor(
        candidate_rhos,
        device=conditions.device,
        dtype=torch.float32,
    )
    conditions[:, :, :sample.real_token_count, :] += (
        envelope
        * rho_tensor.view(1, -1, 1, 1)
        * directions[:, None].float()
    )
    conditions = conditions.flatten(0, 1).to(sample.prompt_embed.dtype)
    candidate_latents = latents[:, None].expand(
        count,
        number_of_candidates,
        *latents.shape[1:],
    ).flatten(0, 1)

    prediction = model.predict_noise(
        pipe,
        candidate_latents,
        timestep,
        sample,
        conditions,
    )
    alpha_bar = pipe.scheduler.alphas_cumprod[int(timestep.item())].to(
        candidate_latents.device,
        torch.float32,
    )
    predicted_x0 = (
        candidate_latents.float()
        - torch.sqrt((1.0 - alpha_bar).clamp_min(0.0)) * prediction.float()
    ) / torch.sqrt(alpha_bar.clamp_min(1e-12))

    images = model.decode_latents(
        pipe,
        predicted_x0.to(candidate_latents.dtype),
    )
    blended = model.blend_images(images, sample)
    crops = [
        metrics.mask_bbox_crop(image, sample.edit_mask_image)
        for image in blended
    ]
    clip_values = metrics.clip_metrics(crops, sample.caption)["clip_all"]
    dino_features = metrics.dino_embeddings(crops, batch_size=count)
    if dino_features is None:
        raise RuntimeError("DINOv2 is required for rho-star selection")

    return {
        "candidate_rhos": candidate_rhos,
        "actual_timestep": int(timestep.item()),
        "schedule_envelope": float(envelope),
        "clip_scores": torch.tensor(clip_values).reshape(
            count, number_of_candidates
        ),
        "dino_features": dino_features.reshape(
            count, number_of_candidates, -1
        ),
        "prefix_seconds": prefix_seconds,
    }


def candidate_rows(probe, selection):
    """Return readable diagnostics for every particle and candidate scale."""
    rhos = torch.tensor(probe["candidate_rhos"], dtype=torch.float32)
    clip_scores = torch.as_tensor(probe["clip_scores"], dtype=torch.float32)
    dino = torch.as_tensor(probe["dino_features"], dtype=torch.float32)
    dino = dino / dino.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    clean_index = int(torch.nonzero(rhos == 0.0).item())
    rows = []
    for particle in range(clip_scores.shape[0]):
        for candidate in range(clip_scores.shape[1]):
            rows.append({
                "particle": particle,
                "rho": float(rhos[candidate]),
                "clip": float(clip_scores[particle, candidate]),
                "clip_change_from_clean": float(
                    clip_scores[particle, candidate]
                    - clip_scores[particle, clean_index]
                ),
                "dino_distance_from_clean": float(
                    1.0 - (
                        dino[particle, candidate]
                        * dino[particle, clean_index]
                    ).sum()
                ),
                "passes_clip_constraint": bool(
                    selection["constraint_feasible_mask"][particle][candidate]
                ),
                "selected": candidate == selection["selected_indexes"][particle],
            })
    return rows


def run(pipe, sample, basis, config, metrics, progress=True):
    """Select one rho per particle, then restart and run adaptive pullback."""
    probe_started = time.perf_counter()
    probe = probe_candidates(pipe, sample, basis, config, metrics)
    selection = select_rho_combination_clip_dino(
        probe,
        max_clip_drop=config.rho_star_max_clip_drop,
        max_combinations=config.rho_star_max_combinations,
        selectable_rhos=list(config.rho_star_candidate_rhos),
        search_strategy=config.rho_star_search_strategy,
        beam_width=config.rho_star_beam_width,
        constraint_fallback=config.rho_star_constraint_fallback,
    )
    probe_seconds = time.perf_counter() - probe_started
    selected_rhos = selection["selected_rhos"]
    print("selected rhos:", [round(value, 4) for value in selected_rhos])

    sampling_started = time.perf_counter()
    raw, blended = pullback.run_adaptive(
        pipe,
        sample,
        basis,
        mode="disjoint",
        rho=selected_rhos,
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
    sampling_seconds = time.perf_counter() - sampling_started
    details = {
        "probe_timestep": probe["actual_timestep"],
        "probe_schedule_envelope": probe["schedule_envelope"],
        "probe_prefix_seconds": probe["prefix_seconds"],
        "probe_seconds": probe_seconds,
        "sampling_seconds": sampling_seconds,
        "candidate_rhos": probe["candidate_rhos"],
        "selected_rhos": selected_rhos,
        "selection": selection,
        "candidate_diagnostics": candidate_rows(probe, selection),
        "selection_region": "edit_mask_bbox",
    }
    return raw, blended, details
