"""Stable Diffusion 1.5 model loading and condition handling."""

import numpy as np
import torch
from PIL import Image


# Model components are loaded once and shared by the sampling methods.
pipe = None
tokenizer = None
text_encoder = None
tokenizer_2 = None
text_encoder_2 = None
unet = None
vae = None
scheduler = None
device = None
model_dtype = None
unet_particle_batch_size = None

# Set by load_model(); "sd15" or "sdxl". SDXL also needs pooled_positive,
# pooled_negative (set by encode_prompt) and add_time_ids_tensor (set by
# make_initial_latents) before predict_epsilon_cfg can run.
model_family = "sdxl"
pooled_positive = None
pooled_negative = None
add_time_ids_tensor = None


def load_model(
    model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    family="sdxl",
    local_files_only=True,
):
    """Load Stable Diffusion 1.5 or SDXL with a DDIM scheduler."""

    global pipe, tokenizer, text_encoder, tokenizer_2, text_encoder_2
    global unet, vae, scheduler, device, model_dtype, model_family

    if family not in {"sd15", "sdxl"}:
        raise ValueError("family must be 'sd15' or 'sdxl'")

    from diffusers import (
        DDIMScheduler,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32

    if family == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            local_files_only=local_files_only,
            variant="fp16",
        ).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            local_files_only=local_files_only,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    tokenizer_2 = getattr(pipe, "tokenizer_2", None)
    text_encoder_2 = getattr(pipe, "text_encoder_2", None)
    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler
    model_family = family

    modules = [text_encoder, unet, vae]
    if text_encoder_2 is not None:
        modules.append(text_encoder_2)
    for module in modules:
        module.eval()
        module.requires_grad_(False)

    return pipe


def require_model():
    if pipe is None:
        raise RuntimeError("Call load_model() before using the pipeline")


def set_unet_particle_batch_size(batch_size):
    """Limit particles per U-Net call. None keeps one full batch."""

    global unet_particle_batch_size
    if batch_size is not None and batch_size < 1:
        raise ValueError("U-Net particle batch size must be positive")
    unet_particle_batch_size = batch_size


def encode_prompt(prompt, negative_prompt=""):
    """Return positive and negative CLIP embeddings and the real-token count."""

    require_model()
    if model_family == "sdxl":
        return _encode_prompt_sdxl(prompt, negative_prompt)
    return _encode_prompt_sd15(prompt, negative_prompt)


def _encode_prompt_sdxl(prompt, negative_prompt):
    """Encode with both SDXL text encoders and capture the pooled embeds."""

    global pooled_positive, pooled_negative

    positive, negative, pooled_positive, pooled_negative = pipe.encode_prompt(
        prompt,
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=negative_prompt,
    )

    tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    eos_position = (tokens.input_ids[0] == tokenizer.eos_token_id).nonzero()[0]
    number_of_real_tokens = int(eos_position.item()) + 1
    return positive, negative, number_of_real_tokens


def _encode_prompt_sd15(prompt, negative_prompt):
    def tokenize(text):
        return tokenizer(
            [text],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

    positive_tokens = tokenize(prompt)
    negative_tokens = tokenize(negative_prompt)
    use_mask = bool(getattr(text_encoder.config, "use_attention_mask", False))

    positive_ids = positive_tokens.input_ids.to(device)
    negative_ids = negative_tokens.input_ids.to(device)
    positive_mask = positive_tokens.attention_mask.to(device)
    negative_mask = negative_tokens.attention_mask.to(device)

    with torch.no_grad():
        positive = text_encoder(
            positive_ids,
            attention_mask=positive_mask if use_mask else None,
        ).last_hidden_state
        negative = text_encoder(
            negative_ids,
            attention_mask=negative_mask if use_mask else None,
        ).last_hidden_state

    number_of_real_tokens = int(positive_mask[0].sum().item())
    return positive, negative, number_of_real_tokens


def make_initial_latents(number_of_particles, height, width, seed):
    """Draw independent initial noises on CPU and then move them to the model."""

    require_model()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    shape = (
        number_of_particles,
        unet.config.in_channels,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    latents = torch.randn(shape, generator=generator, dtype=torch.float32)
    latents = latents.to(device=device, dtype=model_dtype)

    if model_family == "sdxl":
        global add_time_ids_tensor
        add_time_ids_tensor = torch.tensor(
            [list((width, height) + (0, 0) + (width, height))],
            dtype=model_dtype,
            device=device,
        )

    return latents * scheduler.init_noise_sigma


def repeat_condition(condition, batch_size):
    if condition.shape[0] == batch_size:
        return condition
    if condition.shape[0] == 1:
        return condition.repeat(batch_size, *([1] * (condition.dim() - 1)))
    raise ValueError(
        f"Cannot expand condition batch {condition.shape[0]} to {batch_size}"
    )


def predict_epsilon_cfg(
    latents,
    timestep,
    positive_condition,
    negative_condition,
    guidance_scale,
):
    """Predict classifier-free-guided noise in optional microbatches."""

    batch_size = latents.shape[0]
    positive_condition = repeat_condition(positive_condition, batch_size)
    negative_condition = repeat_condition(negative_condition, batch_size)

    sdxl = model_family == "sdxl"
    if sdxl:
        positive_pooled = repeat_condition(pooled_positive, batch_size)
        negative_pooled = repeat_condition(pooled_negative, batch_size)
        time_ids = add_time_ids_tensor.repeat(batch_size, 1)

    particle_batch_size = unet_particle_batch_size or batch_size
    predictions = []
    for start in range(0, batch_size, particle_batch_size):
        stop = min(start + particle_batch_size, batch_size)
        latent_chunk = latents[start:stop]
        positive_chunk = positive_condition[start:stop]
        negative_chunk = negative_condition[start:stop]

        model_latents = torch.cat([latent_chunk, latent_chunk], dim=0)
        model_conditions = torch.cat([negative_chunk, positive_chunk], dim=0)
        model_latents = scheduler.scale_model_input(model_latents, timestep)

        unet_kwargs = {}
        if sdxl:
            pooled_chunk = torch.cat(
                [negative_pooled[start:stop], positive_pooled[start:stop]],
                dim=0,
            )
            time_ids_chunk = time_ids[start:stop].repeat(2, 1)
            unet_kwargs["added_cond_kwargs"] = {
                "text_embeds": pooled_chunk,
                "time_ids": time_ids_chunk,
            }

        with torch.no_grad():
            prediction = unet(
                model_latents,
                timestep,
                encoder_hidden_states=model_conditions,
                **unet_kwargs,
            ).sample

        epsilon_negative, epsilon_positive = prediction.chunk(2)
        predictions.append(
            epsilon_negative
            + guidance_scale * (epsilon_positive - epsilon_negative)
        )

    return torch.cat(predictions, dim=0)


def decode_latents(latents):
    """Decode a latent batch into PIL images.

    Some VAEs (e.g. the stock SDXL base VAE) overflow to NaN when run in
    fp16 and set config.force_upcast=True to signal that they need fp32 to
    decode correctly -- mirrors diffusers' own SDXL pipeline handling.
    """

    needs_upcasting = (
        vae.dtype == torch.float16 and vae.config.force_upcast
    )
    if needs_upcasting:
        vae.to(dtype=torch.float32)
        latents = latents.float()

    with torch.no_grad():
        decoded = vae.decode(latents / vae.config.scaling_factor).sample

    if needs_upcasting:
        vae.to(dtype=model_dtype)

    images = (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    return [
        Image.fromarray((image * 255.0).round().astype(np.uint8))
        for image in images
    ]
