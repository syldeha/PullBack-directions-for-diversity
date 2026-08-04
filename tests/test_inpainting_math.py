"""Small CPU checks for the BrushNet pullback and baseline equations."""

from types import SimpleNamespace

import torch

from inpainting import cads, experiment, model, pullback, tpso
from inpainting.config import InpaintingConfig


def test_regional_pullback_operator():
    generator = torch.Generator().manual_seed(33)
    center = torch.randn((1, 3, 4), generator=generator)
    directions = torch.randn((3, 3, 4), generator=generator)
    jacobian = torch.randn((4, 12), generator=generator)
    anchor = torch.zeros((1, 1, 2, 2))
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    sample = SimpleNamespace(edit_mask_latent=mask)

    def linear_response(pipe, latent, condition, timestep, prepared):
        del pipe, latent, timestep, prepared
        values = condition.float().flatten(1) @ jacobian.T
        return values.reshape(condition.shape[0], 1, 2, 2)

    saved_response = pullback.conditional_noise
    pullback.conditional_noise = linear_response
    try:
        result = pullback.apply_pullback_metric(
            None,
            anchor,
            center,
            directions,
            torch.tensor(500),
            sample,
            response_region="edit_mask",
        )
    finally:
        pullback.conditional_noise = saved_response

    mask_flat = mask.flatten()
    masked_jacobian = mask_flat[:, None] * jacobian
    metric = masked_jacobian.T @ masked_jacobian
    expected = (directions.flatten(1) @ metric).reshape_as(directions)
    torch.testing.assert_close(result, expected, rtol=0.0, atol=2e-5)


def test_fixed_disjoint_directions():
    basis = torch.eye(16, dtype=torch.float32)[:8].reshape(8, 4, 4)
    pipe = SimpleNamespace(_execution_device=torch.device("cpu"))
    sample = SimpleNamespace(prompt_embed=torch.ones((1, 4, 4)))

    first_noise = pullback.make_fixed_prompt_noise(
        pipe,
        sample,
        basis,
        "disjoint",
        777,
        4,
    )
    second_noise = pullback.make_fixed_prompt_noise(
        pipe,
        sample,
        basis,
        "disjoint",
        777,
        4,
    )
    torch.testing.assert_close(first_noise, second_noise, rtol=0.0, atol=0.0)

    directions = pullback.project_fixed_prompt_noise(
        basis,
        first_noise,
        "disjoint",
    ).flatten(1)
    gram = directions @ directions.T
    torch.testing.assert_close(
        gram,
        torch.eye(4),
        rtol=0.0,
        atol=1e-6,
    )


def test_schedules_and_cads_rescale():
    assert pullback.alpha_schedule(999, 999, 500) == 1.0
    assert pullback.alpha_schedule(500, 999, 500) == 0.0
    linear = pullback.alpha_schedule(700, 999, 500)
    assert pullback.schedule_envelope(700, 999, 500, power=2.0) == linear**2

    assert cads.gamma_schedule(999, 900, 600) == 0.0
    assert cads.gamma_schedule(500, 900, 600) == 1.0
    assert cads.gamma_schedule(750, 900, 600) == 0.5

    generator = torch.Generator().manual_seed(5)
    clean = torch.randn((3, 4, 5), generator=generator)
    noisy = torch.randn((3, 4, 5), generator=generator) * 4 + 7
    rescaled = cads.rescale_to_clean(noisy, clean)
    torch.testing.assert_close(
        rescaled.mean(dim=(1, 2)),
        clean.mean(dim=(1, 2)),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        rescaled.std(dim=(1, 2)),
        clean.std(dim=(1, 2)),
        rtol=0.0,
        atol=1e-6,
    )

    scheduler = SimpleNamespace(
        config=SimpleNamespace(num_train_timesteps=1000)
    )
    assert tpso.alpha_schedule(999, scheduler, 0.4) == 1.0
    assert tpso.alpha_schedule(0, scheduler, 0.4) == 0.0


def test_refresh_positions():
    timesteps = torch.tensor([999, 900, 800, 700, 600, 500, 0])
    indices = pullback.adaptive_refresh_indices(
        timesteps,
        start_index=0,
        schedule=(999, 500),
        num_refreshes=2,
        spacing="timestep",
        schedule_power=2.0,
    )
    assert indices == [2, 3]


def test_long_evaluation_uses_independent_noise():
    captured = {}

    def fake_prepare(*args, **kwargs):
        del args
        captured.update(kwargs)
        return "prepared"

    saved_prepare = model.prepare_sample
    model.prepare_sample = fake_prepare
    try:
        config = InpaintingConfig()
        example = SimpleNamespace(
            source_image="image",
            mask="mask",
            prompt="prompt",
        )
        result = experiment.prepare_example("pipe", example, config)
    finally:
        model.prepare_sample = saved_prepare

    assert result == "prepared"
    assert captured["initial_noise"] == "independent"
    assert captured["initial_seed"] == 4242


def main():
    test_regional_pullback_operator()
    test_fixed_disjoint_directions()
    test_schedules_and_cads_rescale()
    test_refresh_positions()
    test_long_evaluation_uses_independent_noise()
    print("inpainting math tests passed")


if __name__ == "__main__":
    main()
