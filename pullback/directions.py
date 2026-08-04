"""Pullback schedules and fixed disjoint prompt directions."""

import torch

from generation import ddim, model


def closest_timestep_index(timesteps, requested):
    values = timesteps.detach().float().cpu()
    return int(torch.argmin((values - float(requested)).abs()).item())


def compute_clean_anchor(
    initial_latent,
    positive_condition,
    negative_condition,
    number_of_steps,
    guidance_scale,
    eta,
    eta_seed,
    requested_timestep,
):
    """Follow clean CFG-DDIM until the state at the requested timestep."""

    model.scheduler.set_timesteps(number_of_steps, device=model.device)
    anchor_index = closest_timestep_index(
        model.scheduler.timesteps,
        requested_timestep,
    )
    eta_generator = ddim.make_device_generator(eta_seed)
    latent = initial_latent.detach().clone()

    for index in range(anchor_index):
        timestep = model.scheduler.timesteps[index]
        epsilon_cfg = model.predict_epsilon_cfg(
            latent,
            timestep,
            positive_condition,
            negative_condition,
            guidance_scale,
        )
        latent = ddim.ddim_step(
            latent,
            timestep,
            epsilon_cfg,
            eta,
            eta_generator,
        )

    actual_timestep = model.scheduler.timesteps[anchor_index]
    return latent.detach(), actual_timestep.detach(), anchor_index


def alpha_schedule(timestep, start, end):
    """Return one early, a linear decay in the window, and zero late."""

    timestep = float(timestep)
    if timestep >= start:
        return 1.0
    if timestep <= end:
        return 0.0
    return (timestep - end) / (start - end)


def schedule_envelope(timestep, start, end, power):
    if start <= end:
        raise ValueError("Pullback schedule start must be larger than end")
    if power <= 0:
        raise ValueError("Pullback schedule power must be positive")
    return alpha_schedule(timestep, start, end) ** float(power)


def snake_slices(rank, number_of_particles):
    """Assign balanced, disjoint basis indexes to the particles."""

    subsets = [[] for _ in range(number_of_particles)]
    for first in range(0, rank, number_of_particles):
        chunk = list(range(first, min(first + number_of_particles, rank)))
        if (first // number_of_particles) % 2 == 1:
            chunk.reverse()
        for particle, basis_index in enumerate(chunk):
            subsets[particle].append(basis_index)
    return subsets


def normalize_particle_directions(directions):
    norms = directions.flatten(start_dim=1).norm(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-8)
    return directions / norms.view(-1, 1, 1)


@torch.no_grad()
def condition_cosine_distances(
    current_condition,
    base_condition,
    number_of_real_tokens,
):
    """Measure real-token condition distances to clean and across particles."""

    current = model.repeat_condition(
        current_condition,
        current_condition.shape[0],
    )[:, :number_of_real_tokens, :].float().flatten(start_dim=1)
    base = model.repeat_condition(
        base_condition,
        current.shape[0],
    )[:, :number_of_real_tokens, :].float().flatten(start_dim=1)

    current = current / current.norm(dim=1, keepdim=True).clamp_min(1e-12)
    base = base / base.norm(dim=1, keepdim=True).clamp_min(1e-12)
    distance_to_base = (
        1.0 - (current * base).sum(dim=1)
    ).clamp(min=0.0, max=2.0)

    indexes = torch.triu_indices(
        current.shape[0],
        current.shape[0],
        offset=1,
        device=current.device,
    )
    similarities = current @ current.T
    pairwise_distance = similarities.new_empty((0,))
    if indexes.shape[1] > 0:
        pairwise_distance = (
            1.0 - similarities[indexes[0], indexes[1]]
        ).clamp(min=0.0, max=2.0)
    return distance_to_base, pairwise_distance


def make_fixed_prompt_noise(basis, mode, seed, number_of_particles):
    """Create one persistent ambient Gaussian field per particle."""

    basis = basis.to(device=model.device, dtype=torch.float32)
    if mode == "disjoint":
        if basis.shape[0] < number_of_particles:
            raise ValueError("Disjoint mode requires rank >= particles")
        subsets = [
            basis[indexes]
            for indexes in snake_slices(basis.shape[0], number_of_particles)
        ]
    elif mode == "random":
        subsets = [basis] * number_of_particles
    else:
        raise ValueError("Pullback mode must be 'random' or 'disjoint'")

    ambient_generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + 104729
    )
    ambient = torch.randn(
        (number_of_particles, *basis.shape[1:]),
        generator=ambient_generator,
        dtype=torch.float32,
    ).to(model.device)

    coefficient_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for particle, subspace in enumerate(subsets):
        target = torch.randn(
            subspace.shape[0],
            generator=coefficient_generator,
            dtype=torch.float32,
        ).to(model.device)
        current = ambient[particle].flatten() @ subspace.flatten(1).T
        ambient[particle] += torch.einsum(
            "k,k...->...",
            target - current,
            subspace,
        )

    return ambient


def project_fixed_prompt_noise(basis, fixed_noise, mode):
    """Project persistent noise into the current random or disjoint basis."""

    basis = basis.to(device=model.device, dtype=torch.float32)
    number_of_particles = fixed_noise.shape[0]
    if mode == "disjoint":
        if basis.shape[0] < number_of_particles:
            raise ValueError("Adaptive disjoint rank must be >= particles")
        subsets = [
            basis[indexes]
            for indexes in snake_slices(basis.shape[0], number_of_particles)
        ]
    elif mode == "random":
        subsets = [basis] * number_of_particles
    else:
        raise ValueError("Pullback mode must be 'random' or 'disjoint'")

    projected = []
    for particle, subspace in enumerate(subsets):
        coordinates = fixed_noise[particle].flatten() @ subspace.flatten(1).T
        direction = torch.einsum("k,k...->...", coordinates, subspace)
        if direction.norm() <= 1e-8:
            direction = subspace[0].clone()
        projected.append(direction)

    return normalize_particle_directions(torch.stack(projected))


def adaptive_refresh_indices(
    timesteps,
    start,
    end,
    number_of_refreshes,
    spacing="timestep",
    schedule_power=1.0,
):
    """Choose unique DDIM indexes inside the active schedule window."""

    if spacing not in {"timestep", "envelope"}:
        raise ValueError("Refresh spacing must be 'timestep' or 'envelope'")

    indices = []
    for refresh in range(1, int(number_of_refreshes) + 1):
        fraction = refresh / float(number_of_refreshes + 1)
        if spacing == "timestep":
            target = start + fraction * (end - start)
        else:
            envelope = 1.0 - fraction
            linear_alpha = envelope ** (1.0 / float(schedule_power))
            target = end + (start - end) * linear_alpha
        index = closest_timestep_index(timesteps, target)
        actual = float(timesteps[index])
        if end < actual < start and index not in indices:
            indices.append(index)
    return sorted(indices)
