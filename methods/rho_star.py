"""Early-probe selection of particle-specific pullback magnitudes."""

from __future__ import annotations

import itertools
import math

import torch

from generation import model
from pullback import directions


def predict_single_condition_branch(
    latents: torch.Tensor,
    timestep: torch.Tensor,
    conditions: torch.Tensor,
    pooled_condition: torch.Tensor | None = None,
):
    """Run one U-Net condition branch with the configured microbatch size."""

    if latents.shape[0] != conditions.shape[0]:
        raise ValueError("Each latent must have one matching condition")
    batch_size = latents.shape[0]
    microbatch = model.unet_particle_batch_size or batch_size
    if model.model_family == "sdxl":
        pooled_full = model.repeat_condition(pooled_condition, batch_size)
        time_ids_full = model.add_time_ids_tensor.repeat(batch_size, 1)
    predictions = []
    for first in range(0, batch_size, microbatch):
        last = min(first + microbatch, batch_size)
        model_input = model.scheduler.scale_model_input(
            latents[first:last], timestep
        )
        unet_kwargs = {}
        if model.model_family == "sdxl":
            unet_kwargs["added_cond_kwargs"] = {
                "text_embeds": pooled_full[first:last],
                "time_ids": time_ids_full[first:last],
            }
        with torch.no_grad():
            prediction = model.unet(
                model_input,
                timestep,
                encoder_hidden_states=conditions[first:last],
                **unet_kwargs,
            ).sample
        predictions.append(prediction)
    return torch.cat(predictions, dim=0)


def make_full_scale_directions(
    basis: torch.Tensor,
    positive_condition: torch.Tensor,
    number_of_real_tokens: int,
    number_of_particles: int,
    mode: str,
    direction_seed: int,
):
    """Reproduce the exact token-relative directions used by the sampler."""

    fixed_noise = directions.make_fixed_prompt_noise(
        basis=basis,
        mode=mode,
        seed=direction_seed,
        number_of_particles=number_of_particles,
    )
    unit_directions = directions.project_fixed_prompt_noise(
        basis=basis,
        fixed_noise=fixed_noise,
        mode=mode,
    )
    token_norms = positive_condition[
        0, :number_of_real_tokens, :
    ].float().norm(dim=-1)
    return unit_directions * token_norms.view(1, -1, 1)


@torch.no_grad()
def probe_rho_candidates(
    latents: torch.Tensor,
    timestep: torch.Tensor,
    positive_condition: torch.Tensor,
    negative_condition: torch.Tensor,
    number_of_real_tokens: int,
    full_scale_directions: torch.Tensor,
    candidate_rhos: list[float],
    guidance_scale: float,
    schedule_start: float,
    schedule_end: float,
    schedule_power: float,
):
    """Measure finite condition response for every particle and candidate rho.

    The probe is entirely internal: it uses the positive/unconditional U-Net
    predictions, CFG guidance vectors, and predicted clean latents. It does not
    use CLIP, DINO, decoded images, or final-generation information.
    """

    if not candidate_rhos:
        raise ValueError("candidate_rhos cannot be empty")
    if 0.0 not in candidate_rhos:
        raise ValueError("candidate_rhos must contain the clean rho=0 control")
    if len(set(float(value) for value in candidate_rhos)) != len(candidate_rhos):
        raise ValueError("candidate_rhos must be unique")
    if any(not math.isfinite(value) or value < 0.0 for value in candidate_rhos):
        raise ValueError("candidate rho values must be finite and non-negative")

    prediction_type = getattr(
        model.scheduler.config,
        "prediction_type",
        "epsilon",
    )
    if prediction_type != "epsilon":
        raise ValueError(
            "The rho-star predicted-x0 probe requires epsilon prediction"
        )

    number_of_particles = latents.shape[0]
    number_of_candidates = len(candidate_rhos)
    expected_direction_shape = (
        number_of_particles,
        number_of_real_tokens,
        positive_condition.shape[-1],
    )
    if full_scale_directions.shape != expected_direction_shape:
        raise ValueError(
            "full_scale_directions has shape "
            f"{tuple(full_scale_directions.shape)}, expected "
            f"{expected_direction_shape}"
        )

    envelope = directions.schedule_envelope(
        float(timestep.item()),
        schedule_start,
        schedule_end,
        schedule_power,
    )
    rhos = torch.tensor(
        candidate_rhos, device=model.device, dtype=torch.float32
    )
    base_positive = model.repeat_condition(
        positive_condition, number_of_particles
    ).float()
    base_negative = model.repeat_condition(
        negative_condition, number_of_particles
    ).to(model.model_dtype)

    candidate_conditions = base_positive[:, None].repeat(
        1, number_of_candidates, 1, 1
    )
    candidate_conditions[
        :, :, :number_of_real_tokens, :
    ] += (
        envelope
        * rhos.view(1, number_of_candidates, 1, 1)
        * full_scale_directions[:, None].float()
    )
    candidate_conditions = candidate_conditions.reshape(
        number_of_particles * number_of_candidates,
        *base_positive.shape[1:],
    ).to(model.model_dtype)
    candidate_latents = latents[:, None].expand(
        number_of_particles,
        number_of_candidates,
        *latents.shape[1:],
    ).reshape(
        number_of_particles * number_of_candidates,
        *latents.shape[1:],
    )

    epsilon_negative = predict_single_condition_branch(
        latents, timestep, base_negative,
        pooled_condition=model.pooled_negative,
    ).float()
    epsilon_positive = predict_single_condition_branch(
        candidate_latents, timestep, candidate_conditions,
        pooled_condition=model.pooled_positive,
    ).float().reshape(
        number_of_particles,
        number_of_candidates,
        *latents.shape[1:],
    )
    guidance = epsilon_positive - epsilon_negative[:, None]
    epsilon_cfg = epsilon_negative[:, None] + float(guidance_scale) * guidance

    timestep_index = int(timestep.item())
    alpha_bar = model.scheduler.alphas_cumprod[timestep_index].float().to(
        model.device
    )
    predicted_x0 = (
        latents[:, None].float()
        - torch.sqrt((1.0 - alpha_bar).clamp_min(0.0)) * epsilon_cfg
    ) / torch.sqrt(alpha_bar.clamp_min(1e-12))

    clean_index = candidate_rhos.index(0.0)
    guidance_flat = guidance.flatten(start_dim=2)
    clean_guidance = guidance_flat[:, clean_index:clean_index + 1]
    guidance_norm = guidance_flat.norm(dim=2).clamp_min(1e-12)
    clean_guidance_norm = clean_guidance.norm(dim=2).clamp_min(1e-12)
    guidance_alignment = (
        guidance_flat * clean_guidance
    ).sum(dim=2) / (guidance_norm * clean_guidance_norm)
    relative_response = (
        guidance_flat - clean_guidance
    ).norm(dim=2) / clean_guidance_norm

    x0_flat = predicted_x0.flatten(start_dim=2)
    x0_flat = x0_flat / x0_flat.norm(dim=2, keepdim=True).clamp_min(1e-12)
    clean_x0 = x0_flat[:, clean_index:clean_index + 1]
    x0_cosine_drift = 1.0 - (x0_flat * clean_x0).sum(dim=2)

    return {
        "candidate_rhos": rhos.cpu(),
        "actual_timestep": timestep.detach().cpu(),
        "schedule_envelope": float(envelope),
        "guidance_alignment": guidance_alignment.clamp(-1.0, 1.0).cpu(),
        "relative_response": relative_response.cpu(),
        "predicted_x0_cosine_drift": x0_cosine_drift.clamp(0.0, 2.0).cpu(),
        "predicted_x0": predicted_x0.float().cpu(),
    }


@torch.no_grad()
def add_clip_dino_probe_features(
    probe: dict,
    prompt: str,
    metrics_calculator,
    decode_batch_size: int = 16,
):
    """Decode probe-time x0 estimates and attach CLIP/DINO measurements.

    The candidate ordering remains particle-major: all rho candidates for
    particle 0, then all candidates for particle 1, and so on. Only the
    one-step Tweedie estimates are decoded; this function does not complete a
    diffusion trajectory for every rho.
    """

    if decode_batch_size < 1:
        raise ValueError("decode_batch_size must be positive")
    predicted_x0 = torch.as_tensor(
        probe["predicted_x0"], dtype=torch.float32
    )
    if predicted_x0.ndim != 5:
        raise ValueError(
            "predicted_x0 must have shape (particles, candidates, C, H, W)"
        )

    number_of_particles, number_of_candidates = predicted_x0.shape[:2]
    flat_x0 = predicted_x0.flatten(0, 1)
    images = []
    for first in range(0, len(flat_x0), decode_batch_size):
        batch = flat_x0[first:first + decode_batch_size].to(
            model.device, dtype=model.model_dtype
        )
        images.extend(model.decode_latents(batch))

    _, clip_values = metrics_calculator.clip_score(images, prompt)
    dino_features = metrics_calculator.dino_features(images).float().cpu()
    clip_scores = torch.as_tensor(clip_values, dtype=torch.float32).reshape(
        number_of_particles, number_of_candidates
    )
    dino_features = dino_features.reshape(
        number_of_particles, number_of_candidates, -1
    )
    dino_features = dino_features / dino_features.norm(
        dim=2, keepdim=True
    ).clamp_min(1e-12)

    enriched = dict(probe)
    enriched["clip_scores"] = clip_scores
    enriched["dino_features"] = dino_features
    return enriched


def mean_pairwise_cosine_distance(normalized_features: torch.Tensor):
    if normalized_features.shape[0] < 2:
        raise ValueError("Rho-star selection requires at least two particles")
    similarities = normalized_features @ normalized_features.T
    indexes = torch.triu_indices(
        len(normalized_features), len(normalized_features), offset=1
    )
    return float((1.0 - similarities[indexes[0], indexes[1]]).mean())


def select_rho_combination(
    probe: dict,
    min_guidance_cosine: float,
    max_relative_response: float | None,
    max_combinations: int = 2_000_000,
    selectable_rhos: list[float] | None = None,
    search_strategy: str = "auto",
    beam_width: int = 4096,
    constraint_fallback: str = "clean",
):
    """Select a diverse feasible predicted-x0 set.

    ``probe`` must contain rho=0 as the clean diagnostic reference. However,
    rho=0 is selectable only when it appears in ``selectable_rhos`` (or when
    ``selectable_rhos`` is None). This separates measurement of fidelity from
    the set of interventions that the search is allowed to choose.
    """

    if not -1.0 <= min_guidance_cosine <= 1.0:
        raise ValueError("min_guidance_cosine must lie in [-1, 1]")
    if max_relative_response is not None and max_relative_response < 0.0:
        raise ValueError("max_relative_response must be non-negative or None")
    if search_strategy not in {"auto", "exact", "beam"}:
        raise ValueError("search_strategy must be 'auto', 'exact', or 'beam'")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if constraint_fallback not in {"clean", "minimum_selectable", "error"}:
        raise ValueError(
            "constraint_fallback must be clean, minimum_selectable, or error"
        )

    rhos = torch.as_tensor(probe["candidate_rhos"], dtype=torch.float32)
    alignment = torch.as_tensor(
        probe["guidance_alignment"], dtype=torch.float32
    )
    response = torch.as_tensor(
        probe["relative_response"], dtype=torch.float32
    )
    predicted_x0 = torch.as_tensor(
        probe["predicted_x0"], dtype=torch.float32
    )
    if alignment.shape != response.shape:
        raise ValueError("Probe alignment and response shapes disagree")
    if alignment.shape[1] != len(rhos):
        raise ValueError("Probe candidate dimension does not match rho values")

    feasible = alignment >= float(min_guidance_cosine)
    if max_relative_response is not None:
        feasible &= response <= float(max_relative_response)

    clean_indexes = torch.nonzero(rhos == 0.0, as_tuple=False).flatten()
    if len(clean_indexes) != 1:
        raise ValueError("Probe must contain exactly one rho=0 candidate")
    clean_index = int(clean_indexes.item())

    if selectable_rhos is None:
        selectable_mask = torch.ones(len(rhos), dtype=torch.bool)
    else:
        if not selectable_rhos:
            raise ValueError("selectable_rhos cannot be empty")
        selectable_mask = torch.zeros(len(rhos), dtype=torch.bool)
        for value in selectable_rhos:
            matches = torch.nonzero(
                torch.isclose(
                    rhos,
                    torch.tensor(float(value)),
                    rtol=1e-6,
                    atol=1e-8,
                ),
                as_tuple=False,
            ).flatten()
            if len(matches) != 1:
                raise ValueError(
                    f"Selectable rho {value} is missing or ambiguous in probe"
                )
            selectable_mask[int(matches.item())] = True

    feasible &= selectable_mask.view(1, -1)
    constraint_feasible = feasible.clone()
    fallback_particles = []
    fallback_candidate_indexes = []
    for particle in range(feasible.shape[0]):
        if feasible[particle].any():
            continue
        if constraint_fallback == "error":
            raise RuntimeError(
                f"Particle {particle} has no selectable rho satisfying the "
                "fidelity constraints"
            )
        if constraint_fallback == "clean":
            fallback_index = clean_index
        else:
            selectable_indexes = torch.nonzero(
                selectable_mask, as_tuple=False
            ).flatten()
            fallback_index = int(
                selectable_indexes[
                    torch.argmin(rhos[selectable_indexes])
                ].item()
            )
        feasible[particle, fallback_index] = True
        fallback_particles.append(particle)
        fallback_candidate_indexes.append(fallback_index)

    feasible_indexes = [
        torch.nonzero(row, as_tuple=False).flatten().tolist()
        for row in feasible
    ]
    number_of_combinations = math.prod(len(row) for row in feasible_indexes)
    resolved_strategy = search_strategy
    if resolved_strategy == "auto":
        resolved_strategy = (
            "exact" if number_of_combinations <= max_combinations else "beam"
        )
    if resolved_strategy == "exact" and number_of_combinations > max_combinations:
        raise ValueError(
            f"Rho search has {number_of_combinations:,} combinations; "
            f"exact-search limit is {max_combinations:,}"
        )

    features = predicted_x0.flatten(start_dim=2)
    features = features / features.norm(dim=2, keepdim=True).clamp_min(1e-12)
    clean_features = features[:, clean_index]
    clean_diversity = mean_pairwise_cosine_distance(clean_features)

    number_of_particles = features.shape[0]
    number_of_pairs = number_of_particles * (number_of_particles - 1) // 2
    pair_distances = {}
    for left in range(number_of_particles):
        for right in range(left + 1, number_of_particles):
            pair_distances[(left, right)] = (
                1.0 - features[left] @ features[right].T
            ).clamp(0.0, 2.0)

    def total_distance(indexes):
        return sum(
            float(pair_distances[(left, right)][indexes[left], indexes[right]])
            for left in range(number_of_particles)
            for right in range(left + 1, number_of_particles)
        )

    states_evaluated = 0
    if resolved_strategy == "exact":
        best_indexes = None
        best_total = -math.inf
        best_rho_sum = math.inf
        for combination in itertools.product(*feasible_indexes):
            states_evaluated += 1
            score = total_distance(combination)
            rho_sum = sum(float(rhos[index]) for index in combination)
            if (
                score > best_total + 1e-12
                or (
                    abs(score - best_total) <= 1e-12
                    and rho_sum < best_rho_sum
                )
            ):
                best_indexes = tuple(combination)
                best_total = score
                best_rho_sum = rho_sum
    else:
        # Each state contains (chosen candidate indexes, accumulated pairwise
        # distance, rho sum). Pairwise candidate distances are precomputed, so
        # the beam search itself is cheap even for high-dimensional latents.
        beam = [(tuple(), 0.0, 0.0)]
        for particle in range(number_of_particles):
            expanded = []
            for indexes, score, rho_sum in beam:
                for candidate in feasible_indexes[particle]:
                    added = sum(
                        float(pair_distances[(previous, particle)][
                            indexes[previous], candidate
                        ])
                        for previous in range(particle)
                    )
                    expanded.append((
                        indexes + (candidate,),
                        score + added,
                        rho_sum + float(rhos[candidate]),
                    ))
            states_evaluated += len(expanded)
            expanded.sort(key=lambda item: (item[1], -item[2]), reverse=True)
            beam = expanded[:beam_width]
        best_indexes, best_total, best_rho_sum = beam[0]

    candidate_indexes = torch.tensor(best_indexes, dtype=torch.long)
    particle_indexes = torch.arange(number_of_particles)
    selected_rhos = rhos[candidate_indexes]
    diversity = best_total / float(number_of_pairs)

    return {
        "selected_indexes": candidate_indexes.tolist(),
        "selected_rhos": selected_rhos.tolist(),
        "predicted_diversity": diversity,
        "clean_predicted_diversity": clean_diversity,
        "predicted_diversity_gain": diversity - clean_diversity,
        "minimum_guidance_alignment": float(
            alignment[particle_indexes, candidate_indexes].min()
        ),
        "maximum_relative_response": float(
            response[particle_indexes, candidate_indexes].max()
        ),
        "mean_rho": float(selected_rhos.mean()),
        "number_of_feasible_combinations": number_of_combinations,
        "search_strategy": resolved_strategy,
        "beam_width": beam_width if resolved_strategy == "beam" else None,
        "search_states_evaluated": states_evaluated,
        "constraint_fallback_particles": fallback_particles,
        "constraint_fallback_candidate_indexes": fallback_candidate_indexes,
        "constraint_fallback_rhos": [
            float(rhos[index]) for index in fallback_candidate_indexes
        ],
        "constraint_feasible_mask": constraint_feasible.tolist(),
        "feasible_mask": feasible.tolist(),
    }


def select_rho_combination_clip_dino(
    probe: dict,
    max_clip_drop: float,
    max_combinations: int = 2_000_000,
    selectable_rhos: list[float] | None = None,
    search_strategy: str = "auto",
    beam_width: int = 4096,
    constraint_fallback: str = "minimum_selectable",
):
    """Maximize probe-time DINO diversity under a one-sided CLIP constraint.

    For particle ``i`` and candidate ``m``, feasibility is

        CLIP(i, m) >= CLIP(i, clean) - max_clip_drop.

    A CLIP improvement is therefore always allowed. Among feasible candidates,
    the joint assignment maximizes mean pairwise DINO cosine distance.
    """

    if max_clip_drop < 0.0 or not math.isfinite(float(max_clip_drop)):
        raise ValueError("max_clip_drop must be finite and non-negative")
    if search_strategy not in {"auto", "exact", "beam"}:
        raise ValueError("search_strategy must be 'auto', 'exact', or 'beam'")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if constraint_fallback not in {"clean", "minimum_selectable", "error"}:
        raise ValueError(
            "constraint_fallback must be clean, minimum_selectable, or error"
        )

    rhos = torch.as_tensor(probe["candidate_rhos"], dtype=torch.float32)
    clip_scores = torch.as_tensor(probe["clip_scores"], dtype=torch.float32)
    dino_features = torch.as_tensor(
        probe["dino_features"], dtype=torch.float32
    )
    if clip_scores.ndim != 2:
        raise ValueError("clip_scores must have shape (particles, candidates)")
    if dino_features.ndim != 3:
        raise ValueError(
            "dino_features must have shape (particles, candidates, features)"
        )
    if dino_features.shape[:2] != clip_scores.shape:
        raise ValueError("CLIP and DINO candidate shapes disagree")
    if clip_scores.shape[1] != len(rhos):
        raise ValueError("Probe candidate dimension does not match rho values")
    if not torch.isfinite(clip_scores).all():
        raise ValueError("clip_scores contains non-finite values")
    if not torch.isfinite(dino_features).all():
        raise ValueError("dino_features contains non-finite values")

    dino_features = dino_features / dino_features.norm(
        dim=2, keepdim=True
    ).clamp_min(1e-12)
    clean_indexes = torch.nonzero(rhos == 0.0, as_tuple=False).flatten()
    if len(clean_indexes) != 1:
        raise ValueError("Probe must contain exactly one rho=0 candidate")
    clean_index = int(clean_indexes.item())

    clean_clip = clip_scores[:, clean_index]
    clip_change = clip_scores - clean_clip[:, None]
    feasible = clip_change >= -float(max_clip_drop)

    if selectable_rhos is None:
        selectable_mask = torch.ones(len(rhos), dtype=torch.bool)
    else:
        if not selectable_rhos:
            raise ValueError("selectable_rhos cannot be empty")
        selectable_mask = torch.zeros(len(rhos), dtype=torch.bool)
        for value in selectable_rhos:
            matches = torch.nonzero(
                torch.isclose(
                    rhos,
                    torch.tensor(float(value)),
                    rtol=1e-6,
                    atol=1e-8,
                ),
                as_tuple=False,
            ).flatten()
            if len(matches) != 1:
                raise ValueError(
                    f"Selectable rho {value} is missing or ambiguous in probe"
                )
            selectable_mask[int(matches.item())] = True

    feasible &= selectable_mask.view(1, -1)
    constraint_feasible = feasible.clone()
    fallback_particles = []
    fallback_candidate_indexes = []
    for particle in range(feasible.shape[0]):
        if feasible[particle].any():
            continue
        if constraint_fallback == "error":
            raise RuntimeError(
                f"Particle {particle} has no selectable rho satisfying the "
                "CLIP constraint"
            )
        if constraint_fallback == "clean":
            fallback_index = clean_index
        else:
            selectable_indexes = torch.nonzero(
                selectable_mask, as_tuple=False
            ).flatten()
            fallback_index = int(
                selectable_indexes[
                    torch.argmin(rhos[selectable_indexes])
                ].item()
            )
        feasible[particle, fallback_index] = True
        fallback_particles.append(particle)
        fallback_candidate_indexes.append(fallback_index)

    feasible_indexes = [
        torch.nonzero(row, as_tuple=False).flatten().tolist()
        for row in feasible
    ]
    number_of_combinations = math.prod(len(row) for row in feasible_indexes)
    resolved_strategy = search_strategy
    if resolved_strategy == "auto":
        resolved_strategy = (
            "exact" if number_of_combinations <= max_combinations else "beam"
        )
    if resolved_strategy == "exact" and number_of_combinations > max_combinations:
        raise ValueError(
            f"Rho search has {number_of_combinations:,} combinations; "
            f"exact-search limit is {max_combinations:,}"
        )

    number_of_particles = dino_features.shape[0]
    if number_of_particles < 2:
        raise ValueError("Rho-star selection requires at least two particles")
    number_of_pairs = number_of_particles * (number_of_particles - 1) // 2
    clean_diversity = mean_pairwise_cosine_distance(
        dino_features[:, clean_index]
    )
    pair_distances = {}
    for left in range(number_of_particles):
        for right in range(left + 1, number_of_particles):
            pair_distances[(left, right)] = (
                1.0 - dino_features[left] @ dino_features[right].T
            ).clamp(0.0, 2.0)

    def total_distance(indexes):
        return sum(
            float(pair_distances[(left, right)][indexes[left], indexes[right]])
            for left in range(number_of_particles)
            for right in range(left + 1, number_of_particles)
        )

    states_evaluated = 0
    if resolved_strategy == "exact":
        best_indexes = None
        best_total = -math.inf
        best_rho_sum = math.inf
        for combination in itertools.product(*feasible_indexes):
            states_evaluated += 1
            score = total_distance(combination)
            rho_sum = sum(float(rhos[index]) for index in combination)
            if (
                score > best_total + 1e-12
                or (
                    abs(score - best_total) <= 1e-12
                    and rho_sum < best_rho_sum
                )
            ):
                best_indexes = tuple(combination)
                best_total = score
                best_rho_sum = rho_sum
    else:
        beam = [(tuple(), 0.0, 0.0)]
        for particle in range(number_of_particles):
            expanded = []
            for indexes, score, rho_sum in beam:
                for candidate in feasible_indexes[particle]:
                    added = sum(
                        float(pair_distances[(previous, particle)][
                            indexes[previous], candidate
                        ])
                        for previous in range(particle)
                    )
                    expanded.append((
                        indexes + (candidate,),
                        score + added,
                        rho_sum + float(rhos[candidate]),
                    ))
            states_evaluated += len(expanded)
            expanded.sort(key=lambda item: (item[1], -item[2]), reverse=True)
            beam = expanded[:beam_width]
        best_indexes, best_total, best_rho_sum = beam[0]

    candidate_indexes = torch.tensor(best_indexes, dtype=torch.long)
    particle_indexes = torch.arange(number_of_particles)
    selected_rhos = rhos[candidate_indexes]
    selected_clip = clip_scores[particle_indexes, candidate_indexes]
    selected_clip_change = clip_change[particle_indexes, candidate_indexes]
    dino_diversity = best_total / float(number_of_pairs)

    return {
        "selection_metric": "dino_cosine_distance",
        "fidelity_constraint": "one_sided_clip_drop",
        "selected_indexes": candidate_indexes.tolist(),
        "selected_rhos": selected_rhos.tolist(),
        "dino_diversity": dino_diversity,
        "clean_dino_diversity": clean_diversity,
        "dino_diversity_gain": dino_diversity - clean_diversity,
        "selected_clip_mean": float(selected_clip.mean()),
        "clean_clip_mean": float(clean_clip.mean()),
        "selected_clip_change_mean": float(selected_clip_change.mean()),
        "selected_clip_change_min": float(selected_clip_change.min()),
        "max_clip_drop": float(max_clip_drop),
        "mean_rho": float(selected_rhos.mean()),
        "number_of_feasible_combinations": number_of_combinations,
        "search_strategy": resolved_strategy,
        "beam_width": beam_width if resolved_strategy == "beam" else None,
        "search_states_evaluated": states_evaluated,
        "constraint_fallback_particles": fallback_particles,
        "constraint_fallback_candidate_indexes": fallback_candidate_indexes,
        "constraint_fallback_rhos": [
            float(rhos[index]) for index in fallback_candidate_indexes
        ],
        "constraint_feasible_mask": constraint_feasible.tolist(),
        "feasible_mask": feasible.tolist(),
    }


def candidate_diagnostic_rows(probe: dict, selection: dict):
    """Flatten candidate diagnostics for CSV/display."""

    rhos = torch.as_tensor(probe["candidate_rhos"])
    alignment = torch.as_tensor(probe["guidance_alignment"])
    response = torch.as_tensor(probe["relative_response"])
    drift = torch.as_tensor(probe["predicted_x0_cosine_drift"])
    feasible = selection["feasible_mask"]
    constraint_feasible = selection.get("constraint_feasible_mask", feasible)
    selected = selection["selected_indexes"]
    clip_scores = probe.get("clip_scores")
    dino_features = probe.get("dino_features")
    clean_index = int(torch.nonzero(rhos == 0.0, as_tuple=False).item())
    if clip_scores is not None:
        clip_scores = torch.as_tensor(clip_scores, dtype=torch.float32)
    if dino_features is not None:
        dino_features = torch.as_tensor(dino_features, dtype=torch.float32)
        dino_features = dino_features / dino_features.norm(
            dim=2, keepdim=True
        ).clamp_min(1e-12)
    rows = []
    for particle in range(alignment.shape[0]):
        for candidate in range(alignment.shape[1]):
            row = {
                "particle": particle,
                "candidate_index": candidate,
                "rho": float(rhos[candidate]),
                "guidance_alignment": float(alignment[particle, candidate]),
                "relative_response": float(response[particle, candidate]),
                "predicted_x0_cosine_drift": float(drift[particle, candidate]),
                "feasible": bool(feasible[particle][candidate]),
                "passes_constraints": bool(
                    constraint_feasible[particle][candidate]
                ),
                "selected": candidate == selected[particle],
            }
            if clip_scores is not None:
                row["clip_score"] = float(clip_scores[particle, candidate])
                row["clip_change_from_clean"] = float(
                    clip_scores[particle, candidate]
                    - clip_scores[particle, clean_index]
                )
            if dino_features is not None:
                row["dino_distance_from_clean"] = float(
                    1.0 - (
                        dino_features[particle, candidate]
                        * dino_features[particle, clean_index]
                    ).sum()
                )
            rows.append(row)
    return rows
