"""TPSO token-prompt optimization for SD1.5 and SDXL BrushNet."""

from dataclasses import dataclass, replace
import time

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from inpainting import model


@dataclass
class EncodedPrompt:
    prompt_embeds: torch.Tensor
    pooled_embeds: torch.Tensor | None
    global_representation: torch.Tensor


@dataclass
class TPSOResult:
    prompt_embeds: torch.Tensor
    pooled_embeds: torch.Tensor | None
    history: list
    offset_norms: list
    steps_run: int
    optimization_seconds: float


def prompt_inputs(pipe, tokenizer, prompt, batch_size):
    """Tokenize one prompt and identify tokens other than BOS/EOS/padding."""
    if hasattr(pipe, "maybe_convert_prompt"):
        prompt = pipe.maybe_convert_prompt(prompt, tokenizer)
    inputs = tokenizer(
        [prompt] * int(batch_size),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(pipe._execution_device)
    attention_mask = inputs.attention_mask.to(pipe._execution_device)

    positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
    lengths = attention_mask.sum(dim=1, keepdim=True)
    content_mask = (positions > 0) & (positions < lengths - 1)
    if not content_mask.any():
        raise ValueError("TPSO needs at least one content token")
    return input_ids, content_mask


def forward_with_token_offset(text_encoder, input_ids, offset, content_mask):
    """Run CLIP while adding a differentiable offset after token lookup."""
    token_layer = text_encoder.text_model.embeddings.token_embedding

    def add_offset(module, inputs, token_embeddings):
        del module, inputs
        active_offset = offset * content_mask.unsqueeze(-1)
        return token_embeddings + active_offset.to(token_embeddings.dtype)

    handle = token_layer.register_forward_hook(add_offset)
    try:
        output = text_encoder(
            input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
    finally:
        handle.remove()
    return output


def prompt_representation(output):
    text_embeds = getattr(output, "text_embeds", None)
    if text_embeds is not None:
        return text_embeds
    return output.pooler_output


def text_components(pipe):
    """Return the one SD1.5 or two SDXL tokenizer/encoder pairs."""
    tokenizers = [pipe.tokenizer]
    text_encoders = [pipe.text_encoder]
    if hasattr(pipe, "tokenizer_2") and hasattr(pipe, "text_encoder_2"):
        tokenizers.append(pipe.tokenizer_2)
        text_encoders.append(pipe.text_encoder_2)
    return tuple(tokenizers), tuple(text_encoders)


def encode_prompt_with_offsets(pipe, prompt, batch_size=1, offsets=None):
    """Encode a clean or offset prompt with the configured text encoders."""
    tokenizers, text_encoders = text_components(pipe)
    if offsets is not None and len(offsets) != len(text_encoders):
        raise ValueError("TPSO expects one offset per text encoder")

    sequence_parts = []
    global_parts = []
    pooled_sdxl = None
    for encoder_index, (tokenizer, text_encoder) in enumerate(
        zip(tokenizers, text_encoders)
    ):
        input_ids, content_mask = prompt_inputs(
            pipe,
            tokenizer,
            prompt,
            batch_size,
        )
        if offsets is None:
            output = text_encoder(
                input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            output = forward_with_token_offset(
                text_encoder,
                input_ids,
                offsets[encoder_index],
                content_mask,
            )
        if len(text_encoders) == 1:
            # SD1.5 conditions the U-Net on the final CLIP hidden state.
            sequence_parts.append(output.last_hidden_state)
        else:
            # SDXL follows the pipeline convention and uses the penultimate state.
            sequence_parts.append(output.hidden_states[-2])
        global_parts.append(
            F.normalize(prompt_representation(output).float(), dim=-1)
        )
        if len(text_encoders) > 1 and encoder_index == len(text_encoders) - 1:
            pooled_sdxl = output.text_embeds

    prompt_embeds = torch.cat(sequence_parts, dim=-1)
    global_representation = F.normalize(
        torch.cat(global_parts, dim=-1),
        dim=-1,
    )
    return EncodedPrompt(
        prompt_embeds,
        pooled_sdxl,
        global_representation,
    )


def make_offsets(pipe, prompt, num_particles, initial_std, seed):
    """Create one learnable token-offset tensor for each text encoder."""
    generator = model.make_generator(pipe._execution_device, seed)
    offsets = []
    masks = []
    tokenizers, text_encoders = text_components(pipe)
    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        input_ids, content_mask = prompt_inputs(
            pipe,
            tokenizer,
            prompt,
            num_particles,
        )
        width = text_encoder.get_input_embeddings().embedding_dim
        value = torch.randn(
            (num_particles, input_ids.shape[1], width),
            generator=generator,
            device=pipe._execution_device,
            dtype=torch.float32,
        ) * float(initial_std)
        value *= content_mask.unsqueeze(-1)
        offsets.append(torch.nn.Parameter(value))
        masks.append(content_mask)
    return offsets, masks


def optimize(
    pipe,
    prompt,
    num_particles,
    kappa=0.80,
    sigma=0.01,
    diversity_weight=1.0,
    learning_rate=1e-3,
    max_steps=200,
    min_steps=50,
    patience=15,
    min_delta=1e-5,
    initial_std=1e-4,
    seed=3407,
    log_every=10,
):
    """Optimize TPSO token offsets while all model weights remain frozen."""
    count = int(num_particles)
    if count < 2:
        raise ValueError("TPSO needs at least two particles")
    if not 0.0 < kappa <= 1.0:
        raise ValueError("kappa must be in (0, 1]")
    if sigma < 0 or max_steps <= 0:
        raise ValueError("sigma and max_steps are invalid")

    _, text_encoders = text_components(pipe)
    for text_encoder in text_encoders:
        text_encoder.eval()
        for parameter in text_encoder.parameters():
            parameter.requires_grad_(False)

    with torch.no_grad():
        clean = encode_prompt_with_offsets(pipe, prompt, batch_size=1)
        clean_representation = clean.global_representation.repeat(count, 1)

    offsets, content_masks = make_offsets(
        pipe,
        prompt,
        count,
        initial_std,
        seed,
    )
    optimizer = torch.optim.Adam(offsets, lr=float(learning_rate))
    off_diagonal = ~torch.eye(
        count,
        device=pipe._execution_device,
        dtype=torch.bool,
    )

    history = []
    stable_steps = 0
    previous_loss = None
    started = time.perf_counter()
    for step in range(int(max_steps)):
        optimizer.zero_grad(set_to_none=True)
        variant = encode_prompt_with_offsets(
            pipe,
            prompt,
            batch_size=count,
            offsets=offsets,
        )

        clean_cosine = (
            variant.global_representation * clean_representation
        ).sum(dim=-1)
        semantic_violation = F.relu(
            (clean_cosine - float(kappa)).abs() - float(sigma)
        )
        semantic_loss = semantic_violation.sum()

        cosine_matrix = (
            variant.global_representation @ variant.global_representation.T
        )
        diversity_loss = cosine_matrix[off_diagonal].mean()
        joint_loss = semantic_loss + float(diversity_weight) * diversity_loss
        joint_loss.backward()
        optimizer.step()

        with torch.no_grad():
            for offset, mask in zip(offsets, content_masks):
                offset.mul_(mask.unsqueeze(-1))

        record = {
            "step": step + 1,
            "joint_loss": float(joint_loss.detach()),
            "semantic_loss": float(semantic_loss.detach()),
            "diversity_loss": float(diversity_loss.detach()),
            "clean_cosine_mean": float(clean_cosine.detach().mean()),
            "clean_cosine_min": float(clean_cosine.detach().min()),
            "pairwise_cosine_mean": float(diversity_loss.detach()),
            "max_semantic_violation": float(semantic_violation.detach().max()),
        }
        history.append(record)

        if log_every and (
            (step + 1) == 1 or (step + 1) % int(log_every) == 0
        ):
            print(
                f"TPSO {step + 1:03d}/{max_steps}: "
                f"joint={record['joint_loss']:.4f} "
                f"clean_cos={record['clean_cosine_mean']:.4f} "
                f"pair_cos={record['pairwise_cosine_mean']:.4f}"
            )

        converged = (
            previous_loss is not None
            and abs(previous_loss - record["joint_loss"]) < float(min_delta)
            and record["max_semantic_violation"] <= float(min_delta)
        )
        stable_steps = stable_steps + 1 if converged else 0
        previous_loss = record["joint_loss"]
        if step + 1 >= int(min_steps) and stable_steps >= int(patience):
            break

    optimization_seconds = time.perf_counter() - started
    with torch.no_grad():
        optimized = encode_prompt_with_offsets(
            pipe,
            prompt,
            batch_size=count,
            offsets=offsets,
        )
        offset_norms = [
            offset.detach().float().flatten(1).norm(dim=1).cpu().tolist()
            for offset in offsets
        ]

    return TPSOResult(
        prompt_embeds=optimized.prompt_embeds.detach(),
        pooled_embeds=(
            optimized.pooled_embeds.detach()
            if optimized.pooled_embeds is not None
            else None
        ),
        history=history,
        offset_norms=offset_norms,
        steps_run=len(history),
        optimization_seconds=optimization_seconds,
    )


def verify_clean_encoding(pipe, sample):
    """Check that zero-offset encoding matches BrushNet prompt encoding."""
    with torch.no_grad():
        clean = encode_prompt_with_offsets(pipe, sample.caption, batch_size=1)
    verification = {
        "prompt_embed_max_error": float(
            (
                clean.prompt_embeds.to(sample.prompt_embed.dtype)
                - sample.prompt_embed
            ).abs().max()
        ),
    }
    if clean.pooled_embeds is None and sample.pooled is None:
        verification["pooled_embed_max_error"] = 0.0
    elif clean.pooled_embeds is None or sample.pooled is None:
        verification["pooled_embed_max_error"] = float("inf")
    else:
        verification["pooled_embed_max_error"] = float(
            (
                clean.pooled_embeds.to(sample.pooled.dtype)
                - sample.pooled
            ).abs().max()
        )
    return verification


def alpha_schedule(timestep, scheduler, ratio):
    """Use optimized prompts early, then linearly return to the clean prompt."""
    ratio = float(ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("TPSO ratio must be in (0, 1]")
    horizon = float(scheduler.config.num_train_timesteps - 1)
    threshold = horizon * (1.0 - ratio)
    return max(
        0.0,
        min(1.0, (float(timestep) - threshold) / (ratio * horizon)),
    )


def predict_noise(
    pipe,
    latents,
    timestep,
    sample,
    prompt_embeds,
    pooled_embeds,
):
    predictions = []
    for particle in range(latents.shape[0]):
        changes = {"prompt_embed": prompt_embeds[particle:particle + 1]}
        if pooled_embeds is not None:
            changes["pooled"] = pooled_embeds[particle:particle + 1]
        particle_sample = replace(sample, **changes)
        model_input, prompt, added, down, middle, up = model.brushnet_forward(
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
            unconditional
            + model.GUIDANCE_SCALE * (conditional - unconditional)
        )
    return torch.cat(predictions)


def ddim_step(
    pipe,
    latents,
    timestep,
    previous_timestep,
    sample,
    eta,
    noise_seed,
    prompt_embeds,
    pooled_embeds,
):
    with torch.no_grad():
        prediction = predict_noise(
            pipe,
            latents,
            timestep,
            sample,
            prompt_embeds,
            pooled_embeds,
        )
    mean, sigma, _ = model.ddim_mean_sigma(
        pipe,
        latents,
        prediction,
        timestep,
        previous_timestep,
        eta,
    )
    noise = torch.randn(
        latents.shape,
        generator=model.make_generator(latents.device, noise_seed),
        device=latents.device,
        dtype=latents.dtype,
    )
    return (mean + sigma * noise).detach()


def run(
    pipe,
    sample,
    optimized,
    num_particles,
    eta,
    noise_seed_base,
    ratio=0.4,
    progress=False,
):
    """Run BrushNet DDIM with TPSO positive conditions and a clean negative."""
    count = int(num_particles)
    if optimized.prompt_embeds.shape[0] != count:
        raise ValueError("TPSO condition batch must match num_particles")
    if sample.pooled is not None and (
        optimized.pooled_embeds is None
        or optimized.pooled_embeds.shape[0] != count
    ):
        raise ValueError("TPSO pooled condition batch must match num_particles")

    clean_prompt = sample.prompt_embed.repeat(count, 1, 1)
    clean_pooled = (
        sample.pooled.repeat(count, 1)
        if sample.pooled is not None
        else None
    )
    latents = sample.initial_latents.clone()
    alpha_trace = []

    iterator = range(sample.start_index, len(sample.timesteps) - 1)
    if progress:
        iterator = tqdm(iterator, leave=False, desc="TPSO sampling")

    for step in iterator:
        alpha = alpha_schedule(
            sample.timesteps[step],
            pipe.scheduler,
            ratio,
        )
        alpha_trace.append((int(sample.timesteps[step]), alpha))
        prompt_t = clean_prompt + alpha * (
            optimized.prompt_embeds.to(clean_prompt.dtype) - clean_prompt
        )
        pooled_t = None
        if clean_pooled is not None:
            pooled_t = clean_pooled + alpha * (
                optimized.pooled_embeds.to(clean_pooled.dtype) - clean_pooled
            )
        latents = ddim_step(
            pipe,
            latents,
            sample.timesteps[step],
            sample.timesteps[step + 1],
            sample,
            eta,
            noise_seed_base + step,
            prompt_t,
            pooled_t,
        )

    raw = model.decode_latents(pipe, latents)
    return raw, model.blend_images(raw, sample), alpha_trace
