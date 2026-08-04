"""Regional condition pullback basis and scheduled BrushNet sampling."""

import torch
from tqdm.auto import tqdm

from inpainting import model


# These are the values used by the audited BrushNet experiments.
PULLBACK_CHUNK = 2
PULLBACK_FD_EPSILON = 0.5
PULLBACK_ITERATIONS = 3


def orthogonalize(directions, maximum=None):
    """Return an orthonormal basis with the same prompt-tensor shape."""
    q, r = torch.linalg.qr(directions.flatten(1).T, mode="reduced")
    q = q[:, torch.diagonal(r).abs() > 1e-7]
    if maximum is not None:
        q = q[:, :maximum]
    if q.shape[1] == 0:
        raise RuntimeError("The pullback directions have zero rank")
    return q.T.reshape(-1, *directions.shape[1:]).contiguous()


def conditional_noise(pipe, anchor_latent, condition, timestep, sample):
    """Evaluate the positive BrushNet denoiser response used by the Jacobian."""
    count = anchor_latent.shape[0]
    model_input = pipe.scheduler.scale_model_input(anchor_latent, timestep)
    brushnet_condition = sample.brushnet_condition.repeat(count, 1, 1, 1)
    pooled = sample.pooled.repeat(count, 1)
    time_ids = sample.add_time_ids.repeat(count, 1)
    added = {"text_embeds": pooled, "time_ids": time_ids}
    down, middle, up = pipe.brushnet(
        model_input,
        timestep,
        encoder_hidden_states=condition,
        brushnet_cond=brushnet_condition,
        conditioning_scale=model.BRUSHNET_SCALE,
        guess_mode=False,
        added_cond_kwargs=added,
        return_dict=False,
    )
    return pipe.unet(
        model_input,
        timestep,
        encoder_hidden_states=condition,
        down_block_add_samples=list(down),
        mid_block_add_sample=middle,
        up_block_add_samples=list(up),
        added_cond_kwargs=added,
        return_dict=False,
    )[0]


def pullback_response_mask(sample, response_region, reference):
    """Return the output-space mask used by the regional metric."""
    if response_region == "global":
        return None
    if response_region != "edit_mask":
        raise ValueError("response_region must be 'global' or 'edit_mask'")
    mask = sample.edit_mask_latent.to(
        device=reference.device,
        dtype=reference.dtype,
    )
    if mask.shape[-2:] != reference.shape[-2:]:
        raise ValueError(
            f"latent mask shape {tuple(mask.shape)} does not match response "
            f"shape {tuple(reference.shape)}"
        )
    return mask


def apply_pullback_metric(
    pipe,
    anchor_latent,
    center_condition,
    directions,
    timestep,
    sample,
    response_region="global",
):
    """Apply J^T J or J^T M^T M J without constructing the Jacobian."""
    center_condition = center_condition.detach()
    differentiable_condition = center_condition.clone().requires_grad_(True)
    with torch.enable_grad():
        base_response = conditional_noise(
            pipe,
            anchor_latent,
            differentiable_condition,
            timestep,
            sample,
        )

    response_mask = pullback_response_mask(
        sample,
        response_region,
        base_response,
    )
    metric_response = (
        base_response
        if response_mask is None
        else base_response * response_mask
    )

    outputs = []
    total = directions.shape[0]
    completed = 0
    for start in range(0, total, PULLBACK_CHUNK):
        block = directions[start:start + PULLBACK_CHUNK]
        displacement = float(PULLBACK_FD_EPSILON) * block.to(
            center_condition.dtype
        )
        expanded_condition = center_condition.expand(block.shape[0], -1, -1)
        expanded_anchor = anchor_latent.expand(block.shape[0], -1, -1, -1)

        with torch.no_grad():
            positive = conditional_noise(
                pipe,
                expanded_anchor,
                expanded_condition + displacement,
                timestep,
                sample,
            )
            negative = conditional_noise(
                pipe,
                expanded_anchor,
                expanded_condition - displacement,
                timestep,
                sample,
            )
            jvp = (positive.float() - negative.float()) / (
                2 * PULLBACK_FD_EPSILON
            )

        metric_jvp = jvp if response_mask is None else jvp * response_mask.float()
        for local_index in range(block.shape[0]):
            completed += 1
            with torch.enable_grad():
                vjp = torch.autograd.grad(
                    metric_response,
                    differentiable_condition,
                    metric_jvp[local_index:local_index + 1].to(
                        metric_response.dtype
                    ),
                    retain_graph=completed < total,
                    create_graph=False,
                )[0]
            outputs.append(vjp.detach().float()[0])

    result = torch.stack(outputs)
    if not torch.isfinite(result).all():
        raise FloatingPointError("The pullback metric returned non-finite values")
    return result


def compute_anchor(pipe, sample, eta, noise_seed_base):
    """Run the first particle with the clean prompt to the basis timestep."""
    latent = sample.initial_latents[0:1].clone()
    condition = sample.prompt_embed.repeat(1, 1, 1)
    for step in range(sample.start_index, sample.basis_index):
        latent = model.ddim_step(
            pipe,
            latent,
            sample.timesteps[step],
            sample.timesteps[step + 1],
            sample,
            eta,
            noise_seed_base + step,
            condition,
        )
    return latent


def compute_basis(
    pipe,
    anchor_latent,
    center_condition,
    timestep,
    sample,
    rank,
    seed=515,
    iterations=PULLBACK_ITERATIONS,
    response_region="global",
):
    """Estimate the top condition-response directions by block power iteration."""
    dimension = int(center_condition[0].numel())
    block_size = min(int(rank), dimension)
    random_directions = torch.randn(
        (block_size, *center_condition.shape[1:]),
        generator=model.make_generator(center_condition.device, seed),
        device=center_condition.device,
        dtype=torch.float32,
    )
    q = orthogonalize(random_directions, block_size)

    for _ in range(int(iterations)):
        metric_q = apply_pullback_metric(
            pipe,
            anchor_latent,
            center_condition,
            q,
            timestep,
            sample,
            response_region=response_region,
        )
        q = orthogonalize(metric_q, block_size)

    metric_q = apply_pullback_metric(
        pipe,
        anchor_latent,
        center_condition,
        q,
        timestep,
        sample,
        response_region=response_region,
    )
    q_flat = q.flatten(1)
    metric_flat = metric_q.flatten(1)
    rayleigh = 0.5 * (
        q_flat @ metric_flat.T + metric_flat @ q_flat.T
    )
    eigenvalues, rotation = torch.linalg.eigh(rayleigh)
    keep = eigenvalues.argsort(descending=True)[: min(rank, q.shape[0])]
    basis = (rotation[:, keep].T @ q_flat).reshape(
        -1,
        *center_condition.shape[1:],
    )
    basis = orthogonalize(basis, len(keep))
    return (
        basis.to(center_condition.device, torch.float32),
        eigenvalues[keep].clamp_min(0).tolist(),
    )


def snake_slices(rank, num_particles):
    """Assign a balanced, disjoint set of basis directions to each particle."""
    assignments = [[] for _ in range(num_particles)]
    for first in range(0, rank, num_particles):
        block = list(range(first, min(first + num_particles, rank)))
        if (first // num_particles) % 2 == 1:
            block = block[::-1]
        for particle, direction in enumerate(block):
            assignments[particle].append(direction)
    return assignments


def build_token_perturbation(
    pipe,
    sample,
    basis,
    mode,
    rho,
    seed,
    num_particles,
):
    """Build fixed random or disjoint positive prompt conditions."""
    device = pipe._execution_device
    conditions = sample.prompt_embed.repeat(num_particles, 1, 1).clone()
    if mode == "none":
        return conditions

    generator = model.make_generator(device, seed)
    if mode == "random":
        coefficients = torch.randn(
            (num_particles, basis.shape[0]),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        directions = torch.einsum(
            "nk,k...->n...",
            coefficients,
            basis.to(device),
        )
    elif mode == "disjoint":
        if basis.shape[0] < num_particles:
            raise ValueError("disjoint mode needs rank >= num_particles")
        values = []
        for indices in snake_slices(basis.shape[0], num_particles):
            subspace = basis[indices].to(device)
            coefficients = torch.randn(
                (subspace.shape[0],),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            values.append(
                torch.einsum("m,m...->...", coefficients, subspace)
            )
        directions = torch.stack(values)
    else:
        raise ValueError("mode must be 'none', 'random', or 'disjoint'")

    directions = normalize_prompt_directions(directions)
    token_norms = sample.prompt_embed[
        0,
        :sample.real_token_count,
        :,
    ].norm(dim=-1).to(sample.prompt_embed.dtype)
    directions = (
        directions * (float(rho) * token_norms).view(1, -1, 1)
    ).to(sample.prompt_embed.dtype)
    conditions[:, :sample.real_token_count, :] += directions
    return conditions


def alpha_schedule(timestep, start, end):
    if timestep >= start:
        return 1.0
    if timestep <= end:
        return 0.0
    return (timestep - end) / (start - end)


def schedule_envelope(timestep, start, end, power=1.0):
    power = float(power)
    if power <= 0:
        raise ValueError("schedule power must be positive")
    return alpha_schedule(timestep, start, end) ** power


def run_scheduled(
    pipe,
    sample,
    perturbed_conditions,
    schedule,
    eta,
    noise_seed_base,
    num_particles,
    progress=False,
    schedule_power=1.0,
):
    """Run DDIM with a constant condition or a return-to-clean schedule."""
    clean = sample.prompt_embed.repeat(num_particles, 1, 1).to(
        perturbed_conditions.dtype
    )
    latents = sample.initial_latents.clone()
    iterator = range(sample.start_index, len(sample.timesteps) - 1)
    if progress:
        iterator = tqdm(iterator, leave=False)

    for step in iterator:
        if schedule is None:
            condition = perturbed_conditions
        else:
            alpha = schedule_envelope(
                float(sample.timesteps[step]),
                *schedule,
                power=schedule_power,
            )
            condition = (
                perturbed_conditions
                if alpha >= 1.0
                else clean + alpha * (perturbed_conditions - clean)
            )
        latents = model.ddim_step(
            pipe,
            latents,
            sample.timesteps[step],
            sample.timesteps[step + 1],
            sample,
            eta,
            noise_seed_base + step,
            condition,
        )

    raw = model.decode_latents(pipe, latents)
    return raw, model.blend_images(raw, sample)


def normalize_prompt_directions(directions):
    norms = directions.flatten(1).norm(dim=1).clamp_min(1e-8).view(-1, 1, 1)
    return directions / norms


def make_fixed_prompt_noise(pipe, sample, basis, mode, seed, num_particles):
    """Create persistent ambient noise for all adaptive basis projections."""
    device = pipe._execution_device
    basis = basis.to(device, torch.float32)
    count = int(num_particles)
    if mode == "disjoint":
        if basis.shape[0] < count:
            raise ValueError("disjoint mode needs rank >= num_particles")
        subspaces = [
            basis[indices]
            for indices in snake_slices(basis.shape[0], count)
        ]
    elif mode == "random":
        subspaces = [basis] * count
    else:
        raise ValueError("mode must be 'random' or 'disjoint'")

    ambient = torch.randn(
        (count, *basis.shape[1:]),
        generator=model.make_generator(device, int(seed) + 104729),
        device=device,
        dtype=torch.float32,
    )
    coefficient_generator = model.make_generator(device, seed)
    for particle, subspace in enumerate(subspaces):
        target = torch.randn(
            (subspace.shape[0],),
            generator=coefficient_generator,
            device=device,
            dtype=torch.float32,
        )
        current = ambient[particle].flatten() @ subspace.flatten(1).T
        ambient[particle] += torch.einsum(
            "m,m...->...",
            target - current,
            subspace,
        )
    return ambient


def project_fixed_prompt_noise(basis, fixed_noise, mode):
    """Project fixed ambient noise into the current random or disjoint basis."""
    device = fixed_noise.device
    basis = basis.to(device, torch.float32)
    count = int(fixed_noise.shape[0])
    if mode == "disjoint":
        if basis.shape[0] < count:
            raise ValueError(
                "adaptive disjoint mode needs rank >= num_particles"
            )
        subspaces = [
            basis[indices]
            for indices in snake_slices(basis.shape[0], count)
        ]
    elif mode == "random":
        subspaces = [basis] * count
    else:
        raise ValueError("mode must be 'random' or 'disjoint'")

    projected = []
    for particle, subspace in enumerate(subspaces):
        coordinates = fixed_noise[particle].flatten() @ subspace.flatten(1).T
        direction = torch.einsum(
            "m,m...->...",
            coordinates,
            subspace,
        )
        if direction.norm() <= 1e-8:
            direction = subspace[0].clone()
        projected.append(direction)
    return normalize_prompt_directions(torch.stack(projected))


def adaptive_refresh_indices(
    timesteps,
    start_index,
    schedule,
    num_refreshes,
    spacing="timestep",
    schedule_power=1.0,
):
    """Map equally spaced refresh targets to unique DDIM indices."""
    count = int(num_refreshes)
    if count <= 0:
        return []
    if schedule is None:
        raise ValueError("adaptive refresh needs a finite schedule")
    start, end = map(float, schedule)
    if start <= end:
        raise ValueError("schedule must satisfy start > end")
    if spacing not in {"timestep", "envelope"}:
        raise ValueError("spacing must be 'timestep' or 'envelope'")

    targets = []
    for refresh in range(1, count + 1):
        fraction = refresh / float(count + 1)
        if spacing == "timestep":
            target = start + fraction * (end - start)
        else:
            envelope = 1.0 - fraction
            linear_alpha = envelope ** (1.0 / float(schedule_power))
            target = end + (start - end) * linear_alpha
        targets.append(target)

    indices = []
    for target in targets:
        index = model.closest_timestep_index(timesteps, target)
        actual_timestep = float(timesteps[index])
        if index <= int(start_index) or index >= len(timesteps) - 1:
            continue
        if not end < actual_timestep < start:
            continue
        if index not in indices:
            indices.append(index)
    return sorted(indices)


def run_adaptive(
    pipe,
    sample,
    basis,
    mode,
    rho,
    num_particles,
    eta,
    noise_seed_base,
    schedule=(999, 500),
    direction_seed=777,
    schedule_power=1.0,
    num_refreshes=2,
    intermediate_rank=8,
    intermediate_iterations=1,
    intermediate_seed=1515,
    refresh_spacing="timestep",
    transition_steps=2,
    anchor_particle=0,
    response_region="global",
    progress=False,
):
    """Run the adaptive scheduled pullback method used in the paper experiments."""
    count = int(num_particles)
    rank = int(intermediate_rank)
    iterations = int(intermediate_iterations)
    transition_steps = int(transition_steps)
    anchor_particle = int(anchor_particle)
    if mode not in {"random", "disjoint"}:
        raise ValueError("mode must be 'random' or 'disjoint'")
    if mode == "disjoint" and rank < count:
        raise ValueError("intermediate_rank must be >= num_particles")
    if rank <= 0 or iterations <= 0:
        raise ValueError("rank and iterations must be positive")
    if transition_steps < 0:
        raise ValueError("transition_steps must be non-negative")
    if not 0 <= anchor_particle < count:
        raise ValueError("anchor_particle is outside the particle batch")

    fixed_noise = make_fixed_prompt_noise(
        pipe,
        sample,
        basis,
        mode,
        direction_seed,
        count,
    )
    active_directions = project_fixed_prompt_noise(basis, fixed_noise, mode)
    source_directions = active_directions
    target_directions = active_directions
    transition_position = transition_steps

    token_norms = sample.prompt_embed[
        0,
        :sample.real_token_count,
        :,
    ].norm(dim=-1).float()
    center = sample.prompt_embed[:, :sample.real_token_count, :].clone()
    clean = sample.prompt_embed.repeat(count, 1, 1)
    clean_float = clean.float()
    latents = sample.initial_latents.clone()

    refresh_indices = adaptive_refresh_indices(
        sample.timesteps,
        sample.start_index,
        schedule,
        num_refreshes,
        spacing=refresh_spacing,
        schedule_power=schedule_power,
    )
    refresh_set = set(refresh_indices)
    if refresh_indices:
        refresh_timesteps = [
            int(sample.timesteps[index].item())
            for index in refresh_indices
        ]
        print(f"adaptive pullback refresh timesteps: {refresh_timesteps}")

    iterator = range(sample.start_index, len(sample.timesteps) - 1)
    if progress:
        iterator = tqdm(iterator, leave=False, desc="adaptive pullback")

    for step in iterator:
        if step in refresh_set:
            actual_timestep = int(sample.timesteps[step].item())
            print(
                f"recomputing rank-{rank} pullback basis at t={actual_timestep} "
                f"from particle {anchor_particle}"
            )
            new_basis, eigenvalues = compute_basis(
                pipe,
                latents[anchor_particle:anchor_particle + 1].detach(),
                center,
                sample.timesteps[step],
                sample,
                rank,
                seed=int(intermediate_seed),
                iterations=iterations,
                response_region=response_region,
            )
            new_directions = project_fixed_prompt_noise(
                new_basis,
                fixed_noise,
                mode,
            )
            source_directions = active_directions.clone()
            target_directions = new_directions
            transition_position = 0
            print(
                "intermediate pullback eigenvalues:",
                [f"{value:.3g}" for value in eigenvalues[:8]],
            )
            torch.cuda.empty_cache()

        if transition_steps == 0 or transition_position >= transition_steps:
            active_directions = target_directions
        else:
            transition_position += 1
            interpolation = transition_position / float(transition_steps)
            active_directions = normalize_prompt_directions(
                (1.0 - interpolation) * source_directions
                + interpolation * target_directions
            )

        alpha = (
            schedule_envelope(
                float(sample.timesteps[step]),
                *schedule,
                power=schedule_power,
            )
            if schedule is not None
            else 1.0
        )
        if alpha <= 0.0:
            condition = clean
        else:
            displacement = active_directions * (
                float(rho) * token_norms
            ).view(1, -1, 1)
            condition_float = clean_float.clone()
            condition_float[:, :sample.real_token_count, :] += (
                alpha * displacement
            )
            condition = condition_float.to(clean.dtype)

        latents = model.ddim_step(
            pipe,
            latents,
            sample.timesteps[step],
            sample.timesteps[step + 1],
            sample,
            eta,
            noise_seed_base + step,
            condition,
        )

    raw = model.decode_latents(pipe, latents)
    return raw, model.blend_images(raw, sample)
