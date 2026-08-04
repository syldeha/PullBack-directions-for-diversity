"""Matrix-free condition pullback operator and top-rank basis estimation."""

import torch

from generation import model


def build_full_positive_condition(
    real_condition,
    base_positive,
    number_of_real_tokens,
):
    """Insert differentiable real-token embeddings into the fixed padding."""

    fixed_padding = base_positive[:, number_of_real_tokens:, :].detach()
    return torch.cat([real_condition, fixed_padding], dim=1)


def positive_noise_map(
    latent,
    timestep,
    real_condition,
    base_positive,
    number_of_real_tokens,
):
    """Map real positive-token embeddings to the U-Net noise prediction."""

    full_condition = build_full_positive_condition(
        real_condition,
        base_positive,
        number_of_real_tokens,
    )
    model_input = model.scheduler.scale_model_input(latent, timestep)
    return model.unet(
        model_input,
        timestep,
        encoder_hidden_states=full_condition,
    ).sample


@torch.no_grad()
def condition_jvp_finite_difference(
    latent,
    timestep,
    center_real_condition,
    direction,
    base_positive,
    number_of_real_tokens,
    finite_difference_epsilon,
):
    """Approximate Jv with a centered finite difference."""

    direction_model_dtype = direction.to(
        device=model.device,
        dtype=center_real_condition.dtype,
    )
    condition_plus = (
        center_real_condition.detach()
        + finite_difference_epsilon * direction_model_dtype
    )
    condition_minus = (
        center_real_condition.detach()
        - finite_difference_epsilon * direction_model_dtype
    )

    prediction_plus = positive_noise_map(
        latent,
        timestep,
        condition_plus,
        base_positive,
        number_of_real_tokens,
    )
    prediction_minus = positive_noise_map(
        latent,
        timestep,
        condition_minus,
        base_positive,
        number_of_real_tokens,
    )
    return (
        prediction_plus.float() - prediction_minus.float()
    ) / (2.0 * finite_difference_epsilon)


def condition_vjp(
    latent,
    timestep,
    center_real_condition,
    output_direction,
    base_positive,
    number_of_real_tokens,
):
    """Compute J transpose u as the condition gradient of the scalar product."""

    differentiable_condition = (
        center_real_condition.detach().clone().requires_grad_(True)
    )
    prediction = positive_noise_map(
        latent,
        timestep,
        differentiable_condition,
        base_positive,
        number_of_real_tokens,
    )
    scalar = (prediction.float() * output_direction.float()).sum()
    gradient = torch.autograd.grad(
        scalar,
        differentiable_condition,
        create_graph=False,
        retain_graph=False,
    )[0]
    return gradient.detach().float()


def pullback_metric_matvec(
    latent,
    timestep,
    center_real_condition,
    direction,
    base_positive,
    number_of_real_tokens,
    finite_difference_epsilon,
):
    """Apply Gv = J transpose Jv without constructing the Jacobian."""

    jvp = condition_jvp_finite_difference(
        latent,
        timestep,
        center_real_condition,
        direction,
        base_positive,
        number_of_real_tokens,
        finite_difference_epsilon,
    )
    return condition_vjp(
        latent,
        timestep,
        center_real_condition,
        jvp,
        base_positive,
        number_of_real_tokens,
    )


def orthonormalize_directions(directions, tolerance=1e-7):
    """QR-orthonormalize directions shaped [rank, tokens, width]."""

    original_shape = directions.shape[1:]
    flat = directions.float().flatten(start_dim=1)
    q, r = torch.linalg.qr(flat.T, mode="reduced")
    valid = torch.diagonal(r).abs() > tolerance
    q = q[:, valid]
    if q.shape[1] == 0:
        raise RuntimeError("Pullback iteration produced zero-rank directions")
    return q.T.reshape(-1, *original_shape).contiguous()


def apply_pullback_metric_block(
    latent,
    timestep,
    center_real_condition,
    directions,
    base_positive,
    number_of_real_tokens,
    finite_difference_epsilon,
):
    """Apply the pullback metric independently to a block of directions."""

    transformed = []
    for index in range(directions.shape[0]):
        transformed.append(
            pullback_metric_matvec(
                latent,
                timestep,
                center_real_condition,
                directions[index:index + 1],
                base_positive,
                number_of_real_tokens,
                finite_difference_epsilon,
            )[0]
        )
    return torch.stack(transformed)


def compute_pullback_basis(
    latent,
    timestep,
    base_positive,
    number_of_real_tokens,
    rank,
    number_of_iterations,
    seed,
    finite_difference_epsilon=0.5,
    progress_label="pullback basis",
):
    """Estimate the dominant eigenvectors with block power iteration."""

    center = base_positive[:, :number_of_real_tokens, :].detach().clone()
    condition_shape = center.shape[1:]
    rank = min(int(rank), int(center[0].numel()))

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    directions = torch.randn(
        (rank, *condition_shape),
        generator=generator,
        dtype=torch.float32,
    ).to(model.device)
    q = orthonormalize_directions(directions)

    for iteration in range(int(number_of_iterations)):
        print(
            f"{progress_label}: power iteration "
            f"{iteration + 1}/{number_of_iterations}"
        )
        metric_q = apply_pullback_metric_block(
            latent,
            timestep,
            center,
            q,
            base_positive,
            number_of_real_tokens,
            finite_difference_epsilon,
        )
        q = orthonormalize_directions(metric_q)

    metric_q = apply_pullback_metric_block(
        latent,
        timestep,
        center,
        q,
        base_positive,
        number_of_real_tokens,
        finite_difference_epsilon,
    )

    q_flat = q.flatten(start_dim=1)
    metric_q_flat = metric_q.flatten(start_dim=1)
    projected_metric = q_flat @ metric_q_flat.T
    projected_metric = 0.5 * (projected_metric + projected_metric.T)

    eigenvalues, eigenvectors_small = torch.linalg.eigh(projected_metric)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors_small = eigenvectors_small[:, order]
    basis_flat = eigenvectors_small.T @ q_flat
    basis = basis_flat.reshape(-1, *condition_shape)
    basis = orthonormalize_directions(basis)

    return basis, eigenvalues.detach().float()


def verify_basis(basis):
    """Return the numerical basis checks used by the experiment runner."""

    flat = basis.float().flatten(start_dim=1)
    gram = flat @ flat.T
    identity = torch.eye(
        gram.shape[0],
        device=gram.device,
        dtype=gram.dtype,
    )
    return {
        "shape": list(basis.shape),
        "finite": bool(torch.isfinite(basis).all()),
        "max_orthogonality_error": float(
            (gram - identity).abs().max().item()
        ),
    }
