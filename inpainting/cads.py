"""CADS conditioning annealing for the BrushNet SDXL pipeline."""

import math

import torch
from tqdm.auto import tqdm

from inpainting import model, pullback


CADS_EXPERIMENTS = [
    {
        "name": "cads_paper_version",
        "mode": "isotropic",
        "negative_mode": "isotropic",
        "persistence": "fresh",
    },
    {
        "name": "cads_isotropic_fixed",
        "mode": "isotropic",
        "negative_mode": "isotropic",
        "persistence": "fixed",
    },
    {
        "name": "cads_random_fresh",
        "mode": "random",
        "negative_mode": "isotropic",
        "persistence": "fresh",
    },
    {
        "name": "cads_random_fixed",
        "mode": "random",
        "negative_mode": "isotropic",
        "persistence": "fixed",
    },
    {
        "name": "cads_disjoint_fresh",
        "mode": "disjoint",
        "negative_mode": "isotropic",
        "persistence": "fresh",
    },
    {
        "name": "cads_disjoint_fixed",
        "mode": "disjoint",
        "negative_mode": "isotropic",
        "persistence": "fixed",
    },
]


def gamma_schedule(timestep, start=900, end=600):
    """Return zero for noisy conditioning and one for the clean condition."""
    timestep = float(timestep)
    if timestep >= start:
        return 0.0
    if timestep <= end:
        return 1.0
    return (start - timestep) / (start - end)


def rescale_to_clean(noisy_condition, clean_condition):
    clean_mean = clean_condition.mean(dim=(1, 2), keepdim=True)
    clean_std = clean_condition.std(dim=(1, 2), keepdim=True)
    noisy_mean = noisy_condition.mean(dim=(1, 2), keepdim=True)
    noisy_std = noisy_condition.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return (
        (noisy_condition - noisy_mean) / noisy_std * clean_std + clean_mean
    )


def raw_isotropic(shape, generator, device):
    return torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def scale_pullback_noise(noise, real_token_count):
    """Match projected noise to the norm of raw CADS noise on real tokens."""
    if real_token_count <= 0:
        raise ValueError("real_token_count must be positive")
    width = noise.shape[-1]
    target_norm = math.sqrt(float(real_token_count * width))
    real_noise = noise[:, :real_token_count, :]
    flat = real_noise.flatten(1)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    scaled = torch.zeros_like(noise)
    scaled[:, :real_token_count, :] = (
        flat.reshape_as(real_noise) * target_norm
    )
    return scaled


def draw_positive_noise(
    mode,
    shape,
    num_particles,
    real_token_count,
    basis,
    subspaces,
    generator,
    device,
):
    if mode == "isotropic":
        return raw_isotropic(shape, generator, device)

    noise = torch.zeros(shape, device=device, dtype=torch.float32)
    if mode == "random":
        coefficients = torch.randn(
            (num_particles, basis.shape[0]),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        noise[:, :real_token_count, :] = torch.einsum(
            "nk,k...->n...",
            coefficients,
            basis,
        )
    elif mode == "disjoint":
        for particle, subspace in enumerate(subspaces):
            coefficients = torch.randn(
                (subspace.shape[0],),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            noise[particle, :real_token_count, :] = torch.einsum(
                "m,m...->...",
                coefficients,
                subspace,
            )
    else:
        raise ValueError("mode must be 'isotropic', 'random', or 'disjoint'")
    return scale_pullback_noise(noise, real_token_count)


def draw_negative_noise(
    negative_mode,
    positive_mode,
    shape,
    num_particles,
    real_token_count,
    basis,
    subspaces,
    generator,
    device,
):
    if negative_mode == "isotropic":
        return raw_isotropic(shape, generator, device)
    if negative_mode == "pullback":
        if positive_mode == "isotropic":
            raise ValueError(
                "negative pullback noise needs random or disjoint positive mode"
            )
        return draw_positive_noise(
            positive_mode,
            shape,
            num_particles,
            real_token_count,
            basis,
            subspaces,
            generator,
            device,
        )
    raise ValueError("negative_mode must be 'isotropic' or 'pullback'")


def prepare_basis(mode, negative_mode, basis, num_particles, device):
    if mode == "isotropic" and negative_mode != "pullback":
        return None, None
    if mode == "isotropic" and negative_mode == "pullback":
        raise ValueError(
            "negative pullback noise needs random or disjoint positive mode"
        )
    if basis is None:
        raise ValueError("this CADS configuration needs a pullback basis")

    basis = basis.to(device=device, dtype=torch.float32)
    subspaces = None
    if mode == "disjoint":
        if basis.shape[0] < num_particles:
            raise ValueError("disjoint mode needs rank >= num_particles")
        subspaces = [
            basis[indices]
            for indices in pullback.snake_slices(
                basis.shape[0],
                num_particles,
            )
        ]
    elif mode != "random":
        raise ValueError("mode must be 'isotropic', 'random', or 'disjoint'")
    return basis, subspaces


def run_cads(
    pipe,
    sample,
    noise_scale,
    num_particles,
    eta,
    noise_seed_base,
    start=900,
    end=600,
    mode="isotropic",
    persistence="fresh",
    negative_mode="isotropic",
    basis=None,
    condition_seed=999,
    rescale_factor=1.0,
    rescale=True,
    progress=False,
):
    """Run CADS with fresh or fixed noise on both CFG branches."""
    device = pipe._execution_device
    if persistence not in {"fresh", "fixed"}:
        raise ValueError("persistence must be 'fresh' or 'fixed'")
    if negative_mode not in {"isotropic", "pullback"}:
        raise ValueError("negative_mode must be 'isotropic' or 'pullback'")

    count = int(num_particles)
    real_token_count = sample.real_token_count
    clean_positive = sample.prompt_embed.repeat(count, 1, 1).float()
    clean_negative = sample.negative_embed.repeat(count, 1, 1).float()
    basis, subspaces = prepare_basis(
        mode,
        negative_mode,
        basis,
        count,
        device,
    )
    generator = model.make_generator(device, condition_seed)

    def positive_noise():
        return draw_positive_noise(
            mode,
            clean_positive.shape,
            count,
            real_token_count,
            basis,
            subspaces,
            generator,
            device,
        )

    def negative_noise():
        return draw_negative_noise(
            negative_mode,
            mode,
            clean_negative.shape,
            count,
            real_token_count,
            basis,
            subspaces,
            generator,
            device,
        )

    fixed_positive = fixed_negative = None
    if persistence == "fixed":
        fixed_positive = positive_noise()
        fixed_negative = negative_noise()

    latents = sample.initial_latents.clone()
    iterator = range(sample.start_index, len(sample.timesteps) - 1)
    if progress:
        iterator = tqdm(iterator, leave=False, desc="CADS")

    for step in iterator:
        gamma = gamma_schedule(
            float(sample.timesteps[step]),
            start=start,
            end=end,
        )
        if gamma >= 1.0:
            positive = clean_positive
            negative = clean_negative
        else:
            positive_draw = (
                fixed_positive if persistence == "fixed" else positive_noise()
            )
            negative_draw = (
                fixed_negative if persistence == "fixed" else negative_noise()
            )
            noise_coefficient = float(noise_scale) * math.sqrt(
                max(0.0, 1.0 - gamma)
            )
            clean_coefficient = math.sqrt(max(0.0, gamma))
            positive = (
                clean_coefficient * clean_positive
                + noise_coefficient * positive_draw
            )
            negative = (
                clean_coefficient * clean_negative
                + noise_coefficient * negative_draw
            )
            if rescale:
                positive_rescaled = rescale_to_clean(
                    positive,
                    clean_positive,
                )
                negative_rescaled = rescale_to_clean(
                    negative,
                    clean_negative,
                )
                positive = (
                    rescale_factor * positive_rescaled
                    + (1.0 - rescale_factor) * positive
                )
                negative = (
                    rescale_factor * negative_rescaled
                    + (1.0 - rescale_factor) * negative
                )

        latents = model.ddim_step_two_branches(
            pipe,
            latents,
            sample.timesteps[step],
            sample.timesteps[step + 1],
            sample,
            eta,
            noise_seed_base + step,
            positive.to(sample.prompt_embed.dtype),
            negative.to(sample.negative_embed.dtype),
        )

    raw = model.decode_latents(pipe, latents)
    return raw, model.blend_images(raw, sample)
