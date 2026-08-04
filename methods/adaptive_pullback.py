"""Scheduled adaptive pullback sampling for Stable Diffusion 1.5."""

import torch
from tqdm.auto import tqdm

from generation import ddim, model
from pullback import basis as pullback_basis
from pullback import directions


def sample_adaptive_pullback(
    initial_latents,
    positive_condition,
    negative_condition,
    number_of_real_tokens,
    initial_basis,
    number_of_steps,
    guidance_scale,
    eta,
    eta_seed,
    rho=1.25,
    start=999,
    end=500,
    schedule_power=2.0,
    mode="disjoint",
    direction_seed=777,
    number_of_refreshes=2,
    intermediate_rank=8,
    intermediate_iterations=1,
    intermediate_seed=1515,
    refresh_spacing="timestep",
    transition_steps=2,
    anchor_particle=0,
    finite_difference_epsilon=0.5,
    progress=True,
    particle_rho=None,
):
    """Run scheduled pullback with low-rank intermediate basis refreshes."""

    number_of_particles = initial_latents.shape[0]
    if not 0 <= anchor_particle < number_of_particles:
        raise ValueError("anchor_particle is outside the particle batch")
    if mode == "disjoint" and intermediate_rank < number_of_particles:
        raise ValueError("Adaptive disjoint rank must be >= particles")

    if particle_rho is None:
        rho_per_particle = torch.full(
            (number_of_particles,),
            float(rho),
            device=model.device,
            dtype=torch.float32,
        )
    else:
        rho_per_particle = torch.as_tensor(
            particle_rho,
            device=model.device,
            dtype=torch.float32,
        ).flatten()
        if rho_per_particle.shape != (number_of_particles,):
            raise ValueError("particle_rho must contain one value per particle")
    if not torch.isfinite(rho_per_particle).all():
        raise ValueError("particle rho values must be finite")
    if (rho_per_particle < 0.0).any():
        raise ValueError("particle rho values must be non-negative")
    has_condition_perturbation = bool((rho_per_particle != 0.0).any())

    model.scheduler.set_timesteps(number_of_steps, device=model.device)
    timesteps = model.scheduler.timesteps
    refresh_indices = directions.adaptive_refresh_indices(
        timesteps,
        start,
        end,
        number_of_refreshes,
        spacing=refresh_spacing,
        schedule_power=schedule_power,
    )
    refresh_set = set(refresh_indices)

    fixed_noise = directions.make_fixed_prompt_noise(
        initial_basis,
        mode,
        direction_seed,
        number_of_particles,
    )
    active_directions = directions.project_fixed_prompt_noise(
        initial_basis,
        fixed_noise,
        mode,
    )
    source_directions = active_directions
    target_directions = active_directions
    transition_position = transition_steps

    base_positive = model.repeat_condition(
        positive_condition,
        number_of_particles,
    )
    base_negative = model.repeat_condition(
        negative_condition,
        number_of_particles,
    )
    token_norms = positive_condition[
        0, :number_of_real_tokens, :
    ].float().norm(dim=-1)

    eta_generator = ddim.make_device_generator(eta_seed)
    latents = initial_latents.detach().clone()
    number_of_pairs = number_of_particles * (number_of_particles - 1) // 2
    condition_base_sum = torch.zeros(
        number_of_particles,
        device=model.device,
        dtype=torch.float32,
    )
    condition_base_active_sum = condition_base_sum.clone()
    condition_base_peak = condition_base_sum.clone()
    condition_pair_sum = torch.zeros(
        number_of_pairs,
        device=model.device,
        dtype=torch.float32,
    )
    condition_pair_active_sum = condition_pair_sum.clone()
    condition_pair_peak = condition_pair_sum.clone()
    condition_trace = []
    condition_active_steps = 0

    iterator = enumerate(timesteps)
    if progress:
        iterator = tqdm(
            iterator,
            total=len(timesteps),
            desc="adaptive pullback",
            leave=False,
        )

    refresh_log = []
    for step_index, timestep in iterator:
        if step_index in refresh_set:
            actual_timestep = int(timestep.item())
            print(
                f"refreshing rank-{intermediate_rank} basis "
                f"at t={actual_timestep}"
            )
            new_basis, eigenvalues = pullback_basis.compute_pullback_basis(
                latent=latents[
                    anchor_particle:anchor_particle + 1
                ].detach(),
                timestep=timestep,
                base_positive=positive_condition,
                number_of_real_tokens=number_of_real_tokens,
                rank=intermediate_rank,
                number_of_iterations=intermediate_iterations,
                seed=intermediate_seed,
                finite_difference_epsilon=finite_difference_epsilon,
                progress_label=f"refresh t={actual_timestep}",
            )
            new_directions = directions.project_fixed_prompt_noise(
                new_basis,
                fixed_noise,
                mode,
            )
            source_directions = active_directions.clone()
            target_directions = new_directions
            transition_position = 0
            refresh_log.append(
                {
                    "timestep": actual_timestep,
                    "eigenvalues": eigenvalues.cpu().tolist(),
                }
            )
            if model.device.type == "cuda":
                torch.cuda.empty_cache()

        if transition_steps == 0 or transition_position >= transition_steps:
            active_directions = target_directions
        else:
            transition_position += 1
            interpolation = transition_position / float(transition_steps)
            active_directions = directions.normalize_particle_directions(
                (1.0 - interpolation) * source_directions
                + interpolation * target_directions
            )

        envelope = directions.schedule_envelope(
            float(timestep),
            start,
            end,
            schedule_power,
        )
        current_positive = base_positive.float().clone()
        if envelope > 0.0 and has_condition_perturbation:
            scaled_direction = active_directions * (
                rho_per_particle.view(-1, 1, 1)
                * token_norms.view(1, -1, 1)
            )
            current_positive[:, :number_of_real_tokens, :] += (
                envelope * scaled_direction
            )

        base_distance, pair_distance = directions.condition_cosine_distances(
            current_positive,
            base_positive,
            number_of_real_tokens,
        )
        condition_base_sum += base_distance
        condition_base_peak = torch.maximum(condition_base_peak, base_distance)
        condition_pair_sum += pair_distance
        condition_pair_peak = torch.maximum(condition_pair_peak, pair_distance)
        if envelope > 0.0:
            condition_active_steps += 1
            condition_base_active_sum += base_distance
            condition_pair_active_sum += pair_distance
        condition_trace.append(
            {
                "timestep": int(timestep.item()),
                "envelope": float(envelope),
                "distance_to_base_mean": float(base_distance.mean().item()),
                "distance_to_base_max": float(base_distance.max().item()),
                "pairwise_distance_mean": (
                    float(pair_distance.mean().item())
                    if pair_distance.numel()
                    else 0.0
                ),
            }
        )

        epsilon_cfg = model.predict_epsilon_cfg(
            latents,
            timestep,
            current_positive.to(model.model_dtype),
            base_negative,
            guidance_scale,
        )
        latents = ddim.ddim_step(
            latents,
            timestep,
            epsilon_cfg,
            eta,
            eta_generator,
        )

    metadata = {
        "refresh_timesteps": [
            int(timesteps[index].item()) for index in refresh_indices
        ],
        "refresh_spacing": refresh_spacing,
        "refreshes": refresh_log,
        "particle_rho": rho_per_particle.cpu().tolist(),
        "condition_cosine": {
            "real_token_count": int(number_of_real_tokens),
            "number_of_steps": int(len(timesteps)),
            "active_steps": int(condition_active_steps),
            "distance_to_base_exposure": (
                condition_base_sum / float(len(timesteps))
            ).cpu().tolist(),
            "distance_to_base_active_mean": (
                condition_base_active_sum
                / float(max(condition_active_steps, 1))
            ).cpu().tolist(),
            "distance_to_base_peak": condition_base_peak.cpu().tolist(),
            "pairwise_distance_exposure": (
                condition_pair_sum / float(len(timesteps))
            ).cpu().tolist(),
            "pairwise_distance_active_mean": (
                condition_pair_active_sum
                / float(max(condition_active_steps, 1))
            ).cpu().tolist(),
            "pairwise_distance_peak": condition_pair_peak.cpu().tolist(),
            "trace": condition_trace,
        },
    }
    return latents, metadata
