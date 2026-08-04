"""Stable Diffusion 1.5 model loading and condition handling."""

import numpy as np
import torch
from PIL import Image


# Model components are loaded once and shared by the sampling methods.
pipe = None
tokenizer = None
text_encoder = None
unet = None
vae = None
scheduler = None
device = None
model_dtype = None
unet_particle_batch_size = None


def load_model(
    model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_files_only=True,
):
    """Load Stable Diffusion 1.5 with its DDIM scheduler."""

    global pipe, tokenizer, text_encoder, unet, vae, scheduler
    global device, model_dtype

    from diffusers import DDIMScheduler, StableDiffusionPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32

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
    unet = pipe.unet
    vae = pipe.vae
    scheduler = pipe.scheduler

    for module in (text_encoder, unet, vae):
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
    return latents * scheduler.init_noise_sigma


def repeat_condition(condition, batch_size):
    if condition.shape[0] == batch_size:
        return condition
    if condition.shape[0] == 1:
        return condition.repeat(batch_size, 1, 1)
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

        with torch.no_grad():
            prediction = unet(
                model_latents,
                timestep,
                encoder_hidden_states=model_conditions,
            ).sample

        epsilon_negative, epsilon_positive = prediction.chunk(2)
        predictions.append(
            epsilon_negative
            + guidance_scale * (epsilon_positive - epsilon_negative)
        )

    return torch.cat(predictions, dim=0)


def decode_latents(latents):
    """Decode a latent batch into PIL images."""

    with torch.no_grad():
        decoded = vae.decode(latents / vae.config.scaling_factor).sample

    images = (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    return [
        Image.fromarray((image * 255.0).round().astype(np.uint8))
        for image in images
    ]
