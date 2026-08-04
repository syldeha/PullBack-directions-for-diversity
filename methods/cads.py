"""Original CADS conditioning-noise baseline."""

import math

import torch

from generation import ddim, model


def cads_gamma(timestep, start=900, end=600):
    """Return the CADS clean-conditioning coefficient."""

    timestep = float(timestep)
    if timestep >= start:
        return 0.0
    if timestep <= end:
        return 1.0
    return (start - timestep) / (start - end)


def rescale_condition(noisy, clean):
    """Match each noisy condition's mean and standard deviation to clean."""

    clean_mean = clean.mean(dim=(1, 2), keepdim=True)
    clean_std = clean.std(dim=(1, 2), keepdim=True)
    noisy_mean = noisy.mean(dim=(1, 2), keepdim=True)
    noisy_std = noisy.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return (noisy - noisy_mean) / noisy_std * clean_std + clean_mean


def sample_cads(
    initial_latents,
    positive_condition,
    negative_condition,
    number_of_steps,
    guidance_scale,
    eta,
    eta_seed,
    start=900,
    end=600,
    noise_scale=0.15,
    psi=1.0,
    noise_seed=999,
    persistence="fresh",
    use_rescale=True,
    progress=True,
):
    """Noise both CFG branches and apply the CADS branch-wise rescaling."""

    if persistence not in {"fresh", "fixed"}:
        raise ValueError("CADS persistence must be 'fresh' or 'fixed'")
    if start <= end:
        raise ValueError("CADS start must be larger than end")

    number_of_particles = initial_latents.shape[0]
    base_positive = model.repeat_condition(
        positive_condition, number_of_particles
    ).float()
    base_negative = model.repeat_condition(
        negative_condition, number_of_particles
    ).float()

    noise_generator = torch.Generator(device="cpu").manual_seed(int(noise_seed))

    def draw_noise(shape):
        return torch.randn(
            shape,
            generator=noise_generator,
            dtype=torch.float32,
        ).to(model.device)

    fixed_positive = fixed_negative = None
    if persistence == "fixed":
        fixed_positive = draw_noise(base_positive.shape)
        fixed_negative = draw_noise(base_negative.shape)

    def condition_provider(step_index, timestep, latents):
        del step_index, latents
        gamma = cads_gamma(float(timestep), start, end)

        if gamma >= 1.0:
            return (
                base_positive.to(model.model_dtype),
                base_negative.to(model.model_dtype),
            )

        positive_noise = (
            fixed_positive
            if persistence == "fixed"
            else draw_noise(base_positive.shape)
        )
        negative_noise = (
            fixed_negative
            if persistence == "fixed"
            else draw_noise(base_negative.shape)
        )

        clean_coefficient = math.sqrt(max(0.0, gamma))
        noise_coefficient = noise_scale * math.sqrt(max(0.0, 1.0 - gamma))

        noisy_positive = (
            clean_coefficient * base_positive
            + noise_coefficient * positive_noise
        )
        noisy_negative = (
            clean_coefficient * base_negative
            + noise_coefficient * negative_noise
        )

        if use_rescale:
            rescaled_positive = rescale_condition(noisy_positive, base_positive)
            rescaled_negative = rescale_condition(noisy_negative, base_negative)
            noisy_positive = psi * rescaled_positive + (1.0 - psi) * noisy_positive
            noisy_negative = psi * rescaled_negative + (1.0 - psi) * noisy_negative

        return (
            noisy_positive.to(model.model_dtype),
            noisy_negative.to(model.model_dtype),
        )

    return ddim.run_ddim_loop(
        initial_latents,
        positive_condition,
        negative_condition,
        number_of_steps,
        guidance_scale,
        eta,
        eta_seed,
        condition_provider=condition_provider,
        progress_label="CADS" if progress else None,
    )
