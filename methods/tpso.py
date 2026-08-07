"""TPSO prompt optimization and scheduled sampling for Stable Diffusion 1.5.

Only one tensor is learned per generated particle: an additive offset on the
content-token embeddings of the positive prompt. The CLIP text encoder and all
diffusion-model parameters remain frozen. Sampling uses the shared DDIM loop,
so TPSO receives the same initial latents, negative prompt, CFG scale, eta,
and sampler seed as every comparison method.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from generation import ddim, model


@dataclass
class TPSOResult:
    positive_conditions: torch.Tensor
    history: list[dict]
    offset_norms: list[float]
    steps_run: int
    optimization_seconds: float
    final_diagnostics: dict


def prompt_inputs(prompt: str, batch_size: int):
    """Tokenize a prompt and mark content tokens, excluding BOS/EOS/padding."""

    model.require_model()
    if hasattr(model.pipe, "maybe_convert_prompt"):
        prompt = model.pipe.maybe_convert_prompt(prompt, model.tokenizer)
    tokens = model.tokenizer(
        [prompt] * int(batch_size),
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(model.device)
    attention_mask = tokens.attention_mask.to(model.device)
    positions = torch.arange(input_ids.shape[1], device=model.device)[None, :]
    lengths = attention_mask.sum(dim=1, keepdim=True)
    content_mask = (positions > 0) & (positions < lengths - 1)
    if not content_mask.any():
        raise ValueError("TPSO requires a prompt with at least one content token")
    return input_ids, attention_mask, content_mask


def prompt_inputs_2(prompt: str, batch_size: int):
    """Tokenize with SDXL's second tokenizer (tokenizer_2 / text_encoder_2)."""

    model.require_model()
    if hasattr(model.pipe, "maybe_convert_prompt"):
        prompt = model.pipe.maybe_convert_prompt(prompt, model.tokenizer_2)
    tokens = model.tokenizer_2(
        [prompt] * int(batch_size),
        padding="max_length",
        max_length=model.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokens.input_ids.to(model.device)
    attention_mask = tokens.attention_mask.to(model.device)
    positions = torch.arange(input_ids.shape[1], device=model.device)[None, :]
    lengths = attention_mask.sum(dim=1, keepdim=True)
    content_mask = (positions > 0) & (positions < lengths - 1)
    if not content_mask.any():
        raise ValueError("TPSO requires a prompt with at least one content token")
    return input_ids, attention_mask, content_mask


def encode_prompt_with_offsets(prompt: str, batch_size: int, offsets=None):
    """Return token conditions and normalized global CLIP features."""

    if model.model_family == "sdxl":
        return _encode_prompt_with_offsets_sdxl(prompt, batch_size, offsets)
    return _encode_prompt_with_offsets_sd15(prompt, batch_size, offsets)


def _encode_prompt_with_offsets_sd15(prompt: str, batch_size: int, offsets=None):
    """Return SD1.5 token conditions and normalized global CLIP features."""

    input_ids, attention_mask, content_mask = prompt_inputs(
        prompt, batch_size
    )
    use_mask = bool(
        getattr(model.text_encoder.config, "use_attention_mask", False)
    )

    handle = None
    if offsets is not None:
        expected = (
            int(batch_size),
            input_ids.shape[1],
            model.text_encoder.get_input_embeddings().embedding_dim,
        )
        if tuple(offsets.shape) != expected:
            raise ValueError(
                f"TPSO offsets have shape {tuple(offsets.shape)}; "
                f"expected {expected}"
            )

        token_layer = model.text_encoder.text_model.embeddings.token_embedding

        def add_offset(_module, _inputs, token_embeddings):
            active = offsets * content_mask.unsqueeze(-1)
            return token_embeddings + active.to(token_embeddings.dtype)

        handle = token_layer.register_forward_hook(add_offset)

    try:
        output = model.text_encoder(
            input_ids,
            attention_mask=attention_mask if use_mask else None,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        if handle is not None:
            handle.remove()

    conditions = output.last_hidden_state
    global_features = F.normalize(output.pooler_output.float(), dim=-1)
    return conditions, global_features, content_mask


def _hook_token_offset(text_encoder, offset_tensor, content_mask, batch_size):
    """Register a forward hook adding a differentiable offset to token embeds.

    Returns the hook handle, or None if offset_tensor is None.
    """

    if offset_tensor is None:
        return None
    expected = (
        int(batch_size),
        content_mask.shape[1],
        text_encoder.get_input_embeddings().embedding_dim,
    )
    if tuple(offset_tensor.shape) != expected:
        raise ValueError(
            f"TPSO offsets have shape {tuple(offset_tensor.shape)}; "
            f"expected {expected}"
        )
    token_layer = text_encoder.text_model.embeddings.token_embedding

    def add_offset(_module, _inputs, token_embeddings):
        active = offset_tensor * content_mask.unsqueeze(-1)
        return token_embeddings + active.to(token_embeddings.dtype)

    return token_layer.register_forward_hook(add_offset)


def _encode_prompt_with_offsets_sdxl(prompt: str, batch_size: int, offsets=None):
    """Return SDXL dual-encoder conditions, pooled embeds, and CLIP features.

    Mirrors StableDiffusionXLPipeline.encode_prompt exactly: both encoders
    run with output_hidden_states=True, the penultimate layer (hidden_states
    [-2]) from each is concatenated for cross-attention conditioning, and the
    "global" feature is text_encoder_2's projected pooled output -- the same
    tensor SDXL uses as add_time_ids-style micro-conditioning.

    offsets, when given, must be a (offsets_1, offsets_2) pair matching the
    two encoders' embedding widths.
    """

    input_ids_1, _, content_mask_1 = prompt_inputs(prompt, batch_size)
    input_ids_2, _, content_mask_2 = prompt_inputs_2(prompt, batch_size)

    offsets_1 = offsets_2 = None
    if offsets is not None:
        offsets_1, offsets_2 = offsets

    handle_1 = _hook_token_offset(
        model.text_encoder, offsets_1, content_mask_1, batch_size
    )
    handle_2 = _hook_token_offset(
        model.text_encoder_2, offsets_2, content_mask_2, batch_size
    )
    try:
        output_1 = model.text_encoder(
            input_ids_1, output_hidden_states=True, return_dict=True
        )
        output_2 = model.text_encoder_2(
            input_ids_2, output_hidden_states=True, return_dict=True
        )
    finally:
        if handle_1 is not None:
            handle_1.remove()
        if handle_2 is not None:
            handle_2.remove()

    conditions = torch.cat(
        [output_1.hidden_states[-2], output_2.hidden_states[-2]], dim=-1
    )
    pooled = output_2.text_embeds
    global_features = F.normalize(pooled.float(), dim=-1)
    return conditions, global_features, pooled, (content_mask_1, content_mask_2)


def verify_clean_encoding(prompt: str, expected_positive: torch.Tensor):
    """Verify that adding a zero token offset leaves the encoding unchanged."""

    if model.model_family == "sdxl":
        return _verify_clean_encoding_sdxl(prompt, expected_positive)
    return _verify_clean_encoding_sd15(prompt, expected_positive)


def _verify_clean_encoding_sd15(prompt: str, expected_positive: torch.Tensor):
    input_ids, _, content_mask = prompt_inputs(prompt, batch_size=1)
    width = model.text_encoder.get_input_embeddings().embedding_dim
    zero_offsets = torch.zeros(
        (1, input_ids.shape[1], width),
        device=model.device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        encoded, _, _ = encode_prompt_with_offsets(
            prompt,
            1,
            offsets=zero_offsets,
        )
    error = float(
        (
            encoded.to(expected_positive.dtype)
            - expected_positive
        ).abs().max()
    )
    return {
        "positive_condition_max_error": error,
        "number_of_optimized_tokens": int(content_mask[0].sum().item()),
    }


def _verify_clean_encoding_sdxl(prompt: str, expected_positive: torch.Tensor):
    input_ids_1, _, content_mask_1 = prompt_inputs(prompt, batch_size=1)
    input_ids_2, _, content_mask_2 = prompt_inputs_2(prompt, batch_size=1)
    width_1 = model.text_encoder.get_input_embeddings().embedding_dim
    width_2 = model.text_encoder_2.get_input_embeddings().embedding_dim
    zero_offsets = (
        torch.zeros(
            (1, input_ids_1.shape[1], width_1),
            device=model.device,
            dtype=torch.float32,
        ),
        torch.zeros(
            (1, input_ids_2.shape[1], width_2),
            device=model.device,
            dtype=torch.float32,
        ),
    )
    with torch.no_grad():
        encoded, _, _, _ = encode_prompt_with_offsets(
            prompt,
            1,
            offsets=zero_offsets,
        )
    error = float(
        (
            encoded.to(expected_positive.dtype)
            - expected_positive
        ).abs().max()
    )
    return {
        "positive_condition_max_error": error,
        "number_of_optimized_tokens": (
            int(content_mask_1[0].sum().item())
            + int(content_mask_2[0].sum().item())
        ),
    }


def semantic_and_diversity(
    clean_features: torch.Tensor,
    variant_features: torch.Tensor,
    kappa: float,
    sigma: float,
):
    """Compute the two TPSO paper losses and useful diagnostics."""

    number_of_particles = variant_features.shape[0]
    clean_cosine = (variant_features * clean_features).sum(dim=-1)
    semantic_violation = F.relu(
        (clean_cosine - float(kappa)).abs() - float(sigma)
    )
    # TPSO deliberately sums this term so every variant keeps its own gradient
    # scale instead of being divided by the number of particles.
    semantic_loss = semantic_violation.sum()

    cosine_matrix = variant_features @ variant_features.T
    off_diagonal = ~torch.eye(
        number_of_particles,
        device=variant_features.device,
        dtype=torch.bool,
    )
    pairwise_cosine = cosine_matrix[off_diagonal]
    diversity_loss = pairwise_cosine.mean()
    diagnostics = {
        "semantic_loss": float(semantic_loss.detach()),
        "diversity_loss": float(diversity_loss.detach()),
        "clean_cosine_mean": float(clean_cosine.detach().mean()),
        "clean_cosine_min": float(clean_cosine.detach().min()),
        "clean_cosine_max": float(clean_cosine.detach().max()),
        "max_semantic_violation": float(
            semantic_violation.detach().max()
        ),
        "pairwise_cosine_mean": float(pairwise_cosine.detach().mean()),
        "pairwise_cosine_min": float(pairwise_cosine.detach().min()),
        "pairwise_cosine_max": float(pairwise_cosine.detach().max()),
    }
    return semantic_loss, diversity_loss, diagnostics


def optimize_token_offsets(
    prompt: str,
    number_of_particles: int,
    kappa: float = 0.80,
    sigma: float = 0.01,
    diversity_weight: float = 1.0,
    learning_rate: float = 1e-3,
    max_steps: int = 200,
    min_steps: int = 50,
    patience: int = 15,
    min_delta: float = 1e-5,
    initialization_std: float = 1e-4,
    seed: int = 3407,
    log_every: int = 10,
):
    """Optimize one positive soft prompt per particle with frozen CLIP."""

    arguments = (
        prompt, number_of_particles, kappa, sigma, diversity_weight,
        learning_rate, max_steps, min_steps, patience, min_delta,
        initialization_std, seed, log_every,
    )
    if model.model_family == "sdxl":
        return _optimize_token_offsets_sdxl(*arguments)
    return _optimize_token_offsets_sd15(*arguments)


def _optimize_token_offsets_sd15(
    prompt: str,
    number_of_particles: int,
    kappa: float = 0.80,
    sigma: float = 0.01,
    diversity_weight: float = 1.0,
    learning_rate: float = 1e-3,
    max_steps: int = 200,
    min_steps: int = 50,
    patience: int = 15,
    min_delta: float = 1e-5,
    initialization_std: float = 1e-4,
    seed: int = 3407,
    log_every: int = 10,
):
    """Optimize one positive soft prompt per particle with frozen CLIP."""

    number_of_particles = int(number_of_particles)
    if number_of_particles < 2:
        raise ValueError("TPSO requires at least two particles")
    if not 0.0 < float(kappa) <= 1.0:
        raise ValueError("TPSO kappa must be in (0, 1]")
    if float(sigma) < 0.0:
        raise ValueError("TPSO sigma must be non-negative")
    if float(diversity_weight) < 0.0:
        raise ValueError("TPSO diversity weight must be non-negative")
    if not 1 <= int(min_steps) <= int(max_steps):
        raise ValueError(
            "TPSO steps must satisfy 1 <= min_steps <= max_steps"
        )
    if float(learning_rate) <= 0.0:
        raise ValueError("TPSO learning rate must be positive")
    if int(patience) < 1:
        raise ValueError("TPSO patience must be positive")
    if float(min_delta) < 0.0:
        raise ValueError("TPSO min_delta must be non-negative")
    if float(initialization_std) < 0.0:
        raise ValueError("TPSO initialization std must be non-negative")

    model.text_encoder.eval()
    model.text_encoder.requires_grad_(False)

    with torch.no_grad():
        _, clean_feature, _ = encode_prompt_with_offsets(
            prompt,
            batch_size=1,
        )
        clean_features = clean_feature.repeat(number_of_particles, 1)

    input_ids, _, content_mask = prompt_inputs(
        prompt, batch_size=number_of_particles
    )
    width = model.text_encoder.get_input_embeddings().embedding_dim
    cpu_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    initial_offsets = torch.randn(
        (number_of_particles, input_ids.shape[1], width),
        generator=cpu_generator,
        dtype=torch.float32,
    ) * float(initialization_std)
    initial_offsets *= content_mask.cpu().unsqueeze(-1)
    offsets = torch.nn.Parameter(initial_offsets.to(model.device))
    optimizer = torch.optim.Adam([offsets], lr=float(learning_rate))

    history = []
    stable_steps = 0
    previous_joint_loss = None
    started = time.perf_counter()

    for step in range(int(max_steps)):
        optimizer.zero_grad(set_to_none=True)
        _, variant_features, _ = encode_prompt_with_offsets(
            prompt, number_of_particles, offsets=offsets
        )
        semantic_loss, diversity_loss, diagnostics = (
            semantic_and_diversity(
                clean_features,
                variant_features,
                kappa,
                sigma,
            )
        )
        joint_loss = (
            semantic_loss
            + float(diversity_weight) * diversity_loss
        )
        joint_loss.backward()
        optimizer.step()

        with torch.no_grad():
            offsets.mul_(content_mask.unsqueeze(-1))

        record = {
            "step": step + 1,
            "joint_loss": float(joint_loss.detach()),
            **diagnostics,
        }
        history.append(record)
        if log_every and (
            step == 0 or (step + 1) % int(log_every) == 0
        ):
            print(
                f"TPSO {step + 1:03d}/{max_steps}: "
                f"joint={record['joint_loss']:.4f} "
                f"clean_cos={record['clean_cosine_mean']:.4f} "
                f"pair_cos={record['pairwise_cosine_mean']:.4f}",
                flush=True,
            )

        converged = (
            previous_joint_loss is not None
            and abs(previous_joint_loss - record["joint_loss"])
            < float(min_delta)
            and record["max_semantic_violation"] <= float(min_delta)
        )
        stable_steps = stable_steps + 1 if converged else 0
        previous_joint_loss = record["joint_loss"]
        if (
            step + 1 >= int(min_steps)
            and stable_steps >= int(patience)
        ):
            break

    optimization_seconds = time.perf_counter() - started
    with torch.no_grad():
        optimized_conditions, optimized_features, _ = encode_prompt_with_offsets(
            prompt, number_of_particles, offsets=offsets
        )
        semantic_loss, diversity_loss, final_diagnostics = (
            semantic_and_diversity(
                clean_features,
                optimized_features,
                kappa,
                sigma,
            )
        )
        final_diagnostics["joint_loss"] = float(
            semantic_loss + float(diversity_weight) * diversity_loss
        )
        offset_norms = offsets.float().flatten(1).norm(dim=1).cpu().tolist()

    return TPSOResult(
        positive_conditions=optimized_conditions.detach(),
        history=history,
        offset_norms=offset_norms,
        steps_run=len(history),
        optimization_seconds=optimization_seconds,
        final_diagnostics=final_diagnostics,
    )


def _optimize_token_offsets_sdxl(
    prompt: str,
    number_of_particles: int,
    kappa: float = 0.80,
    sigma: float = 0.01,
    diversity_weight: float = 1.0,
    learning_rate: float = 1e-3,
    max_steps: int = 200,
    min_steps: int = 50,
    patience: int = 15,
    min_delta: float = 1e-5,
    initialization_std: float = 1e-4,
    seed: int = 3407,
    log_every: int = 10,
):
    """SDXL variant: jointly optimizes offsets on both text encoders.

    The "global feature" used for the semantic/diversity losses is
    text_encoder_2's projected pooled output -- the same tensor SDXL treats
    as its own global text representation, so this is a principled reuse
    rather than a separate, TPSO-only feature.
    """

    number_of_particles = int(number_of_particles)
    if number_of_particles < 2:
        raise ValueError("TPSO requires at least two particles")
    if not 0.0 < float(kappa) <= 1.0:
        raise ValueError("TPSO kappa must be in (0, 1]")
    if float(sigma) < 0.0:
        raise ValueError("TPSO sigma must be non-negative")
    if float(diversity_weight) < 0.0:
        raise ValueError("TPSO diversity weight must be non-negative")
    if not 1 <= int(min_steps) <= int(max_steps):
        raise ValueError(
            "TPSO steps must satisfy 1 <= min_steps <= max_steps"
        )
    if float(learning_rate) <= 0.0:
        raise ValueError("TPSO learning rate must be positive")
    if int(patience) < 1:
        raise ValueError("TPSO patience must be positive")
    if float(min_delta) < 0.0:
        raise ValueError("TPSO min_delta must be non-negative")
    if float(initialization_std) < 0.0:
        raise ValueError("TPSO initialization std must be non-negative")

    model.text_encoder.eval()
    model.text_encoder.requires_grad_(False)
    model.text_encoder_2.eval()
    model.text_encoder_2.requires_grad_(False)

    with torch.no_grad():
        _, clean_feature, _, _ = encode_prompt_with_offsets(
            prompt,
            batch_size=1,
        )
        clean_features = clean_feature.repeat(number_of_particles, 1)

    input_ids_1, _, content_mask_1 = prompt_inputs(
        prompt, batch_size=number_of_particles
    )
    input_ids_2, _, content_mask_2 = prompt_inputs_2(
        prompt, batch_size=number_of_particles
    )
    width_1 = model.text_encoder.get_input_embeddings().embedding_dim
    width_2 = model.text_encoder_2.get_input_embeddings().embedding_dim

    cpu_generator = torch.Generator(device="cpu").manual_seed(int(seed))
    initial_offsets_1 = torch.randn(
        (number_of_particles, input_ids_1.shape[1], width_1),
        generator=cpu_generator,
        dtype=torch.float32,
    ) * float(initialization_std)
    initial_offsets_1 *= content_mask_1.cpu().unsqueeze(-1)
    initial_offsets_2 = torch.randn(
        (number_of_particles, input_ids_2.shape[1], width_2),
        generator=cpu_generator,
        dtype=torch.float32,
    ) * float(initialization_std)
    initial_offsets_2 *= content_mask_2.cpu().unsqueeze(-1)

    offsets_1 = torch.nn.Parameter(initial_offsets_1.to(model.device))
    offsets_2 = torch.nn.Parameter(initial_offsets_2.to(model.device))
    optimizer = torch.optim.Adam(
        [offsets_1, offsets_2], lr=float(learning_rate)
    )

    history = []
    stable_steps = 0
    previous_joint_loss = None
    started = time.perf_counter()

    for step in range(int(max_steps)):
        optimizer.zero_grad(set_to_none=True)
        _, variant_features, _, _ = encode_prompt_with_offsets(
            prompt, number_of_particles, offsets=(offsets_1, offsets_2)
        )
        semantic_loss, diversity_loss, diagnostics = (
            semantic_and_diversity(
                clean_features,
                variant_features,
                kappa,
                sigma,
            )
        )
        joint_loss = (
            semantic_loss
            + float(diversity_weight) * diversity_loss
        )
        joint_loss.backward()
        optimizer.step()

        with torch.no_grad():
            offsets_1.mul_(content_mask_1.unsqueeze(-1))
            offsets_2.mul_(content_mask_2.unsqueeze(-1))

        record = {
            "step": step + 1,
            "joint_loss": float(joint_loss.detach()),
            **diagnostics,
        }
        history.append(record)
        if log_every and (
            step == 0 or (step + 1) % int(log_every) == 0
        ):
            print(
                f"TPSO {step + 1:03d}/{max_steps}: "
                f"joint={record['joint_loss']:.4f} "
                f"clean_cos={record['clean_cosine_mean']:.4f} "
                f"pair_cos={record['pairwise_cosine_mean']:.4f}",
                flush=True,
            )

        converged = (
            previous_joint_loss is not None
            and abs(previous_joint_loss - record["joint_loss"])
            < float(min_delta)
            and record["max_semantic_violation"] <= float(min_delta)
        )
        stable_steps = stable_steps + 1 if converged else 0
        previous_joint_loss = record["joint_loss"]
        if (
            step + 1 >= int(min_steps)
            and stable_steps >= int(patience)
        ):
            break

    optimization_seconds = time.perf_counter() - started
    with torch.no_grad():
        optimized_conditions, optimized_features, _, _ = (
            encode_prompt_with_offsets(
                prompt, number_of_particles, offsets=(offsets_1, offsets_2)
            )
        )
        semantic_loss, diversity_loss, final_diagnostics = (
            semantic_and_diversity(
                clean_features,
                optimized_features,
                kappa,
                sigma,
            )
        )
        final_diagnostics["joint_loss"] = float(
            semantic_loss + float(diversity_weight) * diversity_loss
        )
        offset_norms = (
            offsets_1.float().flatten(1).norm(dim=1)
            + offsets_2.float().flatten(1).norm(dim=1)
        ).cpu().tolist()

    return TPSOResult(
        positive_conditions=optimized_conditions.detach(),
        history=history,
        offset_norms=offset_norms,
        steps_run=len(history),
        optimization_seconds=optimization_seconds,
        final_diagnostics=final_diagnostics,
    )


def schedule_alpha(timestep, ratio: float):
    """TPSO schedule: optimized at the start, linearly clean after ratio*T."""

    ratio = float(ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("TPSO ratio must be in (0, 1]")
    if torch.is_tensor(timestep):
        timestep = timestep.item()
    horizon = float(model.scheduler.config.num_train_timesteps - 1)
    threshold = horizon * (1.0 - ratio)
    return max(
        0.0,
        min(1.0, (float(timestep) - threshold) / (ratio * horizon)),
    )


def sample_tpso(
    initial_latents: torch.Tensor,
    clean_positive: torch.Tensor,
    negative_condition: torch.Tensor,
    optimized: TPSOResult,
    number_of_steps: int,
    guidance_scale: float,
    eta: float,
    eta_seed: int,
    ratio: float = 0.4,
    progress: bool = True,
):
    """Sample with optimized positive conditions and a clean negative branch."""

    number_of_particles = initial_latents.shape[0]
    clean_positive = model.repeat_condition(
        clean_positive, number_of_particles
    )
    if optimized.positive_conditions.shape[0] != number_of_particles:
        raise ValueError("TPSO condition batch must match the particle batch")

    optimized_positive = optimized.positive_conditions.to(
        device=model.device,
        dtype=clean_positive.dtype,
    )
    alpha_trace = []

    def condition_provider(step_index, timestep, latents):
        del step_index, latents
        alpha = schedule_alpha(timestep, ratio)
        alpha_trace.append((int(timestep.item()), alpha))
        positive = clean_positive + alpha * (
            optimized_positive - clean_positive
        )
        return positive, negative_condition

    final = ddim.run_ddim_loop(
        initial_latents=initial_latents,
        positive_condition=clean_positive,
        negative_condition=negative_condition,
        number_of_steps=number_of_steps,
        guidance_scale=guidance_scale,
        eta=eta,
        eta_seed=eta_seed,
        condition_provider=condition_provider,
        progress_label="TPSO" if progress else None,
    )
    return final, alpha_trace
