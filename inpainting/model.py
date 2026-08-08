"""BrushNet SD1.5/SDXL loading, conditioning, and DDIM utilities."""

from dataclasses import dataclass, replace
from pathlib import Path
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


# The upstream BrushNet checkout and model weights stay outside this repository.
BRUSHNET_ROOT = Path(
    os.environ.get(
        "BRUSHNET_ROOT",
        Path.home() / "BrushNet",
    )
)
if str(BRUSHNET_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(BRUSHNET_ROOT / "src"))

from diffusers import BrushNetModel, DDIMScheduler
from diffusers.pipelines.brushnet.pipeline_brushnet import (
    StableDiffusionBrushNetPipeline,
)
from diffusers.pipelines.brushnet.pipeline_brushnet_sd_xl import (
    StableDiffusionXLBrushNetPipeline,
)


MODEL_FAMILY = os.environ.get("BRUSHNET_MODEL_FAMILY", "sd15").lower()
if MODEL_FAMILY not in {"sd15", "sdxl"}:
    raise ValueError("BRUSHNET_MODEL_FAMILY must be 'sd15' or 'sdxl'")

DEFAULT_BASE_MODEL = (
    "stable-diffusion-v1-5/stable-diffusion-v1-5"
    if MODEL_FAMILY == "sd15"
    else str(BRUSHNET_ROOT / "checkpoints" / "JuggernautXL-v9")
)
DEFAULT_BRUSHNET_MODEL = (
    BRUSHNET_ROOT / "checkpoints" / "brushnet_segmentation_mask"
    if MODEL_FAMILY == "sd15"
    else BRUSHNET_ROOT / "checkpoints" / "brushnet_sdxl"
)
BASE_MODEL = os.environ.get("BRUSHNET_BASE_MODEL", DEFAULT_BASE_MODEL)
BRUSHNET_MODEL = Path(
    os.environ.get(
        "BRUSHNET_MODEL",
        DEFAULT_BRUSHNET_MODEL,
    )
)
MODEL_DTYPE = torch.float16
NEGATIVE_PROMPT = "low quality, blurry, distorted, artifacts"
GUIDANCE_SCALE = 7.5
BRUSHNET_SCALE = 1.0


@dataclass
class PreparedSample:
    """All shared inputs used by every method for one BrushBench example."""

    prompt_embed: torch.Tensor
    negative_embed: torch.Tensor
    pooled: torch.Tensor
    negative_pooled: torch.Tensor
    add_time_ids: torch.Tensor
    brushnet_condition: torch.Tensor
    edit_mask_latent: torch.Tensor
    initial_latents: torch.Tensor
    timesteps: torch.Tensor
    start_index: int
    basis_index: int
    real_token_count: int
    source_image: Image.Image
    edit_mask_image: Image.Image
    caption: str


def model_paths():
    """Resolve the external BrushNet checkout and checkpoint paths."""
    brushnet_root = Path(os.environ.get("BRUSHNET_ROOT", BRUSHNET_ROOT))
    default_base = (
        "stable-diffusion-v1-5/stable-diffusion-v1-5"
        if MODEL_FAMILY == "sd15"
        else str(brushnet_root / "checkpoints" / "JuggernautXL-v9")
    )
    base_model = os.environ.get("BRUSHNET_BASE_MODEL", default_base)
    default_brushnet = (
        brushnet_root / "checkpoints" / "brushnet_segmentation_mask"
        if MODEL_FAMILY == "sd15"
        else brushnet_root / "checkpoints" / "brushnet_sdxl"
    )
    brushnet_model = Path(
        os.environ.get(
            "BRUSHNET_MODEL",
            default_brushnet,
        )
    )
    return brushnet_root, base_model, brushnet_model


def load_pipeline():
    """Load the configured SD1.5 or SDXL backbone and BrushNet checkpoint."""
    brushnet_root, base_model, brushnet_model = model_paths()
    required_files = [
        (brushnet_root / "src" / "diffusers" / "__init__.py", "BRUSHNET_ROOT"),
        (brushnet_model / "config.json", "BRUSHNET_MODEL"),
    ]
    base_path = Path(base_model)
    if base_path.exists() and not (base_path / "model_index.json").is_file():
        required_files.append(
            (base_path / "model_index.json", "BRUSHNET_BASE_MODEL")
        )
    missing = [str(path) for path, _ in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "BrushNet checkpoints were not found. Set BRUSHNET_ROOT, "
            "BRUSHNET_MODEL, and BRUSHNET_BASE_MODEL before importing this "
            "module. Missing files: " + ", ".join(missing)
        )

    brushnet_arguments = {
        "torch_dtype": MODEL_DTYPE,
        "use_safetensors": True,
    }
    if MODEL_FAMILY == "sd15":
        brushnet_arguments["variant"] = "fp16"
    brushnet = BrushNetModel.from_pretrained(
        brushnet_model,
        **brushnet_arguments,
    )

    pipeline_class = (
        StableDiffusionBrushNetPipeline
        if MODEL_FAMILY == "sd15"
        else StableDiffusionXLBrushNetPipeline
    )
    pipeline_arguments = {
        "brushnet": brushnet,
        "torch_dtype": MODEL_DTYPE,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if MODEL_FAMILY == "sdxl":
        pipeline_arguments["variant"] = "fp16"
    else:
        pipeline_arguments.update({
            "safety_checker": None,
            "requires_safety_checker": False,
        })
    pipe = pipeline_class.from_pretrained(
        base_model,
        **pipeline_arguments,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    modules = [pipe.vae, pipe.text_encoder, pipe.unet, pipe.brushnet]
    if MODEL_FAMILY == "sdxl":
        modules.append(pipe.text_encoder_2)
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return pipe


def decode_rle_mask(mask_rle, shape):
    """Decode a BrushBench mask using the official BrushNet convention."""
    starts = np.asarray(mask_rle[0:][::2], dtype=int)
    lengths = np.asarray(mask_rle[1:][::2], dtype=int)
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask.reshape(shape)


def target_size(image, resolution):
    width, height = image.size
    scale = float(resolution) / float(min(width, height))
    width = max(64, int(round(width * scale / 64)) * 64)
    height = max(64, int(round(height * scale / 64)) * 64)
    return width, height


def make_generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


def closest_timestep_index(timesteps, timestep):
    return int(
        torch.argmin((timesteps.float() - float(timestep)).abs()).item()
    )


def encode_image_latent(pipe, image):
    """Encode an image into the VAE's latent space.

    Some VAEs (e.g. the stock SDXL base VAE) overflow to NaN when run in
    fp16 in either direction -- mirrors the upcast handling in
    decode_latents.
    """
    device = pipe._execution_device
    needs_upcasting = (
        pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    )
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float32)

    pixels = pipe.image_processor.preprocess(
        image,
        height=image.height,
        width=image.width,
    ).to(device, torch.float32 if needs_upcasting else pipe.vae.dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(pixels).latent_dist.mode()

    if needs_upcasting:
        pipe.vae.to(dtype=MODEL_DTYPE)
        latent = latent.to(MODEL_DTYPE)

    return latent * pipe.vae.config.scaling_factor


def prepare_sample(
    pipe,
    source_image,
    mask,
    prompt,
    num_particles=4,
    resolution=1024,
    ddim_steps=50,
    start_timestep=999,
    basis_timestep=500,
    initial_noise="independent",
    initial_seed=4242,
    jitter_std=0.005,
):
    """Prepare one image, mask, prompt, and common initial particle batch."""
    device = pipe._execution_device
    width, height = target_size(source_image, resolution)
    source = source_image.resize((width, height), Image.Resampling.LANCZOS)
    mask_image = Image.fromarray(
        (mask * 255).astype(np.uint8),
        mode="L",
    ).resize((width, height), Image.Resampling.NEAREST)
    mask_values = (np.asarray(mask_image) > 127).astype(np.uint8) * 255
    edit_mask = Image.fromarray(mask_values, mode="L")
    masked_source = Image.composite(
        Image.new("RGB", (width, height), "black"),
        source,
        edit_mask,
    )

    if MODEL_FAMILY == "sd15":
        prompt_embed, negative_embed = pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=NEGATIVE_PROMPT,
        )
        pooled = None
        negative_pooled = None
        time_ids = None
    else:
        prompt_embed, negative_embed, pooled, negative_pooled = pipe.encode_prompt(
            prompt,
            prompt,
            device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=NEGATIVE_PROMPT,
        )
        time_ids = torch.tensor(
            [list((width, height) + (0, 0) + (width, height))],
            dtype=prompt_embed.dtype,
            device=device,
        )

    condition_latent = encode_image_latent(pipe, masked_source)
    mask_pixels = torch.from_numpy(
        (np.asarray(edit_mask, np.float32) / 255)[None, None]
    ).to(device, pipe.vae.dtype)
    edit_mask_latent = F.interpolate(
        mask_pixels,
        size=condition_latent.shape[-2:],
        mode="nearest",
    )
    brushnet_condition = torch.cat(
        [condition_latent, 1 - edit_mask_latent],
        dim=1,
    )

    pipe.scheduler.set_timesteps(ddim_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    start_index = closest_timestep_index(timesteps, start_timestep)
    basis_index = closest_timestep_index(timesteps, basis_timestep)

    latent_shape = tuple(condition_latent.shape[1:])
    if initial_noise == "shared_jitter":
        shared = torch.randn(
            latent_shape,
            generator=make_generator(device, initial_seed),
            device=device,
            dtype=condition_latent.dtype,
        )
        jitter = torch.randn(
            (num_particles, *latent_shape),
            generator=make_generator(device, initial_seed + 1),
            device=device,
            dtype=condition_latent.dtype,
        ) * jitter_std
        initial_latents = shared.unsqueeze(0) + jitter
    elif initial_noise == "independent":
        initial_latents = torch.randn(
            (num_particles, *latent_shape),
            generator=make_generator(device, initial_seed),
            device=device,
            dtype=condition_latent.dtype,
        )
    else:
        raise ValueError(
            "initial_noise must be 'independent' or 'shared_jitter'"
        )

    tokens = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    eos_position = (tokens.input_ids[0] == pipe.tokenizer.eos_token_id).nonzero()[0]
    real_token_count = int(eos_position.item()) + 1

    return PreparedSample(
        prompt_embed,
        negative_embed,
        pooled,
        negative_pooled,
        time_ids,
        brushnet_condition,
        edit_mask_latent,
        initial_latents,
        timesteps,
        start_index,
        basis_index,
        real_token_count,
        source,
        edit_mask,
        prompt,
    )


def repeat_condition(sample, count, classifier_free_guidance):
    condition = sample.brushnet_condition.repeat(count, 1, 1, 1)
    prompt = sample.prompt_embed.repeat(count, 1, 1)
    if sample.pooled is None:
        if not classifier_free_guidance:
            return prompt, condition, None
        negative = sample.negative_embed.repeat(count, 1, 1)
        return (
            torch.cat([negative, prompt]),
            torch.cat([condition, condition]),
            None,
        )

    pooled = sample.pooled.repeat(count, 1)
    time_ids = sample.add_time_ids.repeat(count, 1)
    if not classifier_free_guidance:
        return prompt, condition, {"text_embeds": pooled, "time_ids": time_ids}

    negative = sample.negative_embed.repeat(count, 1, 1)
    negative_pooled = sample.negative_pooled.repeat(count, 1)
    negative_time_ids = sample.add_time_ids.repeat(count, 1)
    return (
        torch.cat([negative, prompt]),
        torch.cat([condition, condition]),
        {
            "text_embeds": torch.cat([negative_pooled, pooled]),
            "time_ids": torch.cat([negative_time_ids, time_ids]),
        },
    )


def brushnet_forward(pipe, latents, timestep, sample, classifier_free_guidance):
    model_input = (
        torch.cat([latents, latents])
        if classifier_free_guidance
        else latents
    )
    model_input = pipe.scheduler.scale_model_input(model_input, timestep)
    prompt, condition, added = repeat_condition(
        sample,
        latents.shape[0],
        classifier_free_guidance,
    )
    down, middle, up = pipe.brushnet(
        model_input,
        timestep,
        encoder_hidden_states=prompt,
        brushnet_cond=condition,
        conditioning_scale=BRUSHNET_SCALE,
        guess_mode=False,
        added_cond_kwargs=added,
        return_dict=False,
    )
    return model_input, prompt, added, tuple(down), middle, tuple(up)


def predict_noise(pipe, latents, timestep, sample, positive_conditions):
    """CFG prediction with one positive token condition per particle."""
    predictions = []
    for particle in range(latents.shape[0]):
        particle_sample = replace(
            sample,
            prompt_embed=positive_conditions[particle:particle + 1],
        )
        model_input, prompt, added, down, middle, up = brushnet_forward(
            pipe,
            latents[particle:particle + 1],
            timestep,
            particle_sample,
            classifier_free_guidance=True,
        )
        output = pipe.unet(
            model_input,
            timestep,
            encoder_hidden_states=prompt,
            down_block_add_samples=list(down),
            mid_block_add_sample=middle,
            up_block_add_samples=list(up),
            added_cond_kwargs=added,
            return_dict=False,
        )[0]
        unconditional, conditional = output.chunk(2)
        predictions.append(
            unconditional + GUIDANCE_SCALE * (conditional - unconditional)
        )
    return torch.cat(predictions)


def predict_noise_two_branches(
    pipe,
    latents,
    timestep,
    sample,
    positive_conditions,
    negative_conditions,
):
    """CFG prediction with particle-specific positive and negative tokens."""
    predictions = []
    for particle in range(latents.shape[0]):
        particle_sample = replace(
            sample,
            prompt_embed=positive_conditions[particle:particle + 1],
            negative_embed=negative_conditions[particle:particle + 1],
        )
        model_input, prompt, added, down, middle, up = brushnet_forward(
            pipe,
            latents[particle:particle + 1],
            timestep,
            particle_sample,
            classifier_free_guidance=True,
        )
        output = pipe.unet(
            model_input,
            timestep,
            encoder_hidden_states=prompt,
            down_block_add_samples=list(down),
            mid_block_add_sample=middle,
            up_block_add_samples=list(up),
            added_cond_kwargs=added,
            return_dict=False,
        )[0]
        unconditional, conditional = output.chunk(2)
        predictions.append(
            unconditional + GUIDANCE_SCALE * (conditional - unconditional)
        )
    return torch.cat(predictions)


def ddim_mean_sigma(pipe, latents, noise_prediction, timestep, previous_timestep, eta):
    alpha_t = pipe.scheduler.alphas_cumprod[int(timestep.item())].to(
        latents.device,
        latents.dtype,
    )
    alpha_previous = pipe.scheduler.alphas_cumprod[
        int(previous_timestep.item())
    ].to(latents.device, latents.dtype)
    clean = (
        latents - (1 - alpha_t).sqrt() * noise_prediction
    ) / alpha_t.sqrt()
    variance = (
        (1 - alpha_previous)
        / (1 - alpha_t)
        * (1 - alpha_t / alpha_previous)
    )
    sigma = eta * variance.clamp_min(0).sqrt()
    epsilon_direction = (
        latents - alpha_t.sqrt() * clean
    ) / (1 - alpha_t).sqrt()
    mean = (
        alpha_previous.sqrt() * clean
        + (1 - alpha_previous - sigma**2).clamp_min(0).sqrt()
        * epsilon_direction
    )
    return mean.detach(), sigma.detach(), clean.detach()


def ddim_step(
    pipe,
    latents,
    timestep,
    previous_timestep,
    sample,
    eta,
    noise_seed,
    positive_conditions,
):
    with torch.no_grad():
        prediction = predict_noise(
            pipe,
            latents,
            timestep,
            sample,
            positive_conditions,
        )
    mean, sigma, _ = ddim_mean_sigma(
        pipe,
        latents,
        prediction,
        timestep,
        previous_timestep,
        eta,
    )
    noise = torch.randn(
        latents.shape,
        generator=make_generator(latents.device, noise_seed),
        device=latents.device,
        dtype=latents.dtype,
    )
    return (mean + sigma * noise).detach()


def ddim_step_two_branches(
    pipe,
    latents,
    timestep,
    previous_timestep,
    sample,
    eta,
    noise_seed,
    positive_conditions,
    negative_conditions,
):
    with torch.no_grad():
        prediction = predict_noise_two_branches(
            pipe,
            latents,
            timestep,
            sample,
            positive_conditions,
            negative_conditions,
        )
    mean, sigma, _ = ddim_mean_sigma(
        pipe,
        latents,
        prediction,
        timestep,
        previous_timestep,
        eta,
    )
    noise = torch.randn(
        latents.shape,
        generator=make_generator(latents.device, noise_seed),
        device=latents.device,
        dtype=latents.dtype,
    )
    return (mean + sigma * noise).detach()


def decode_latents(pipe, latents):
    """Decode a latent batch into PIL images.

    Some VAEs (e.g. the stock SDXL base VAE) overflow to NaN when run in
    fp16 and set config.force_upcast=True to signal that they need fp32 to
    decode correctly -- mirrors diffusers' own SDXL pipeline handling.
    """

    needs_upcasting = (
        pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    )
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float32)

    images = []
    with torch.no_grad():
        for particle in range(latents.shape[0]):
            particle_latents = latents[particle:particle + 1]
            if needs_upcasting:
                particle_latents = particle_latents.float()
            decoded = pipe.vae.decode(
                particle_latents / pipe.vae.config.scaling_factor,
                return_dict=False,
            )[0]
            images.extend(
                pipe.image_processor.postprocess(decoded, output_type="pil")
            )

    if needs_upcasting:
        pipe.vae.to(dtype=MODEL_DTYPE)

    torch.cuda.empty_cache()
    return images


def blend_images(images, sample, blur=6):
    mask = sample.edit_mask_image.filter(ImageFilter.GaussianBlur(blur))
    return [Image.composite(image, sample.source_image, mask) for image in images]
