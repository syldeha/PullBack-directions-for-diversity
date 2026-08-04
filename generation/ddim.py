"""Shared classifier-free-guided DDIM sampling loop."""

import torch
from tqdm.auto import tqdm

from generation import model


def make_device_generator(seed):
    return torch.Generator(device=model.device.type).manual_seed(int(seed))


def ddim_step(latents, timestep, epsilon_prediction, eta, generator):
    """Apply one scheduler transition to the original, unscaled latent."""

    return model.scheduler.step(
        model_output=epsilon_prediction,
        timestep=timestep,
        sample=latents,
        eta=float(eta),
        generator=generator,
        return_dict=True,
    ).prev_sample


def run_ddim_loop(
    initial_latents,
    positive_condition,
    negative_condition,
    number_of_steps,
    guidance_scale,
    eta,
    eta_seed,
    condition_provider=None,
    progress_label=None,
):
    """Run the shared reverse process with optional time-dependent conditions."""

    model.scheduler.set_timesteps(number_of_steps, device=model.device)
    latents = initial_latents.detach().clone()
    eta_generator = make_device_generator(eta_seed)

    iterator = model.scheduler.timesteps
    if progress_label:
        iterator = tqdm(iterator, desc=progress_label, leave=False)

    for step_index, timestep in enumerate(iterator):
        if condition_provider is None:
            current_positive = positive_condition
            current_negative = negative_condition
        else:
            current_positive, current_negative = condition_provider(
                step_index, timestep, latents
            )

        epsilon_cfg = model.predict_epsilon_cfg(
            latents,
            timestep,
            current_positive,
            current_negative,
            guidance_scale,
        )
        latents = ddim_step(
            latents,
            timestep,
            epsilon_cfg,
            eta,
            eta_generator,
        )

    return latents


def sample_clean_ddim(
    initial_latents,
    positive_condition,
    negative_condition,
    number_of_steps,
    guidance_scale,
    eta,
    eta_seed,
    progress=True,
):
    """Sample the clean DDIM baseline."""

    return run_ddim_loop(
        initial_latents,
        positive_condition,
        negative_condition,
        number_of_steps,
        guidance_scale,
        eta,
        eta_seed,
        progress_label="clean DDIM" if progress else None,
    )
