"""Small CPU checks for the generation sampler and pullback basis."""

from types import SimpleNamespace

import torch

from generation import ddim, model
from methods import cads
from methods.adaptive_pullback import sample_adaptive_pullback
from pullback import basis as pullback_basis
from pullback import directions


class FakeScheduler:
    def __init__(self):
        self.config = SimpleNamespace(
            prediction_type="epsilon",
            num_train_timesteps=1000,
        )
        self.alphas_cumprod = torch.linspace(0.99, 0.20, 1000)
        self.init_noise_sigma = 1.0
        self.timesteps = None

    def set_timesteps(self, number_of_steps, device=None):
        self.timesteps = torch.linspace(
            999,
            0,
            int(number_of_steps),
            device=device,
        ).round().long()

    @staticmethod
    def scale_model_input(latents, timestep):
        return latents * (1.0 + float(timestep) / 10000.0)

    @staticmethod
    def step(
        model_output,
        timestep,
        sample,
        eta,
        generator,
        return_dict,
    ):
        del timestep, eta, generator, return_dict
        return SimpleNamespace(prev_sample=sample - 0.075 * model_output)


class FakeUNet:
    def __call__(self, latents, timestep, encoder_hidden_states):
        condition = encoder_hidden_states.float()
        first = condition[:, 0, 0].view(-1, 1, 1, 1)
        second = condition[:, 1, 1].view(-1, 1, 1, 1)
        response = first + 0.3 * second.square()
        response = 0.21 * latents.float() + response + float(timestep) / 10000.0
        return SimpleNamespace(sample=response.to(latents.dtype))


def configure_fake_model():
    model.device = torch.device("cpu")
    model.model_dtype = torch.float32
    model.scheduler = FakeScheduler()
    model.unet = FakeUNet()
    model.unet_particle_batch_size = 2


def inputs():
    initial = torch.tensor([
        [[[0.2, -0.1], [0.3, 0.4]]],
        [[[-0.3, 0.5], [0.1, -0.2]]],
    ])
    positive = torch.tensor([[[1.0, 0.2], [0.3, 0.8], [0.1, 0.1]]])
    negative = torch.zeros_like(positive)
    basis = torch.tensor([
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [0.0, 0.0]],
    ])
    return initial, positive, negative, basis


def test_clean_ddim_known_result():
    configure_fake_model()
    initial, positive, negative, _ = inputs()
    result = ddim.sample_clean_ddim(
        initial,
        positive,
        negative,
        number_of_steps=5,
        guidance_scale=7.5,
        eta=0.0,
        eta_seed=20800,
        progress=False,
    )
    expected = torch.tensor([
        [[[-3.0798521, -3.3558533], [-2.9878516, -2.8958509]]],
        [[[-3.5398540, -2.8038502], [-3.1718524, -3.4478540]]],
    ])
    torch.testing.assert_close(result, expected, rtol=0.0, atol=2e-6)


def test_cads_noise_modes_are_deterministic():
    initial, positive, negative, _ = inputs()
    results = {}
    for persistence in ("fresh", "fixed"):
        configure_fake_model()
        first = cads.sample_cads(
            initial,
            positive,
            negative,
            number_of_steps=5,
            guidance_scale=7.5,
            eta=0.0,
            eta_seed=20800,
            start=900,
            end=600,
            noise_scale=0.15,
            psi=1.0,
            noise_seed=999,
            persistence=persistence,
            use_rescale=True,
            progress=False,
        )
        configure_fake_model()
        second = cads.sample_cads(
            initial,
            positive,
            negative,
            number_of_steps=5,
            guidance_scale=7.5,
            eta=0.0,
            eta_seed=20800,
            start=900,
            end=600,
            noise_scale=0.15,
            psi=1.0,
            noise_seed=999,
            persistence=persistence,
            use_rescale=True,
            progress=False,
        )
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        results[persistence] = first

    assert not torch.equal(results["fresh"], results["fixed"])


def test_adaptive_pullback_is_deterministic():
    initial, positive, negative, basis = inputs()
    outputs = []
    metadata = []
    for _ in range(2):
        configure_fake_model()
        result, details = sample_adaptive_pullback(
            initial,
            positive,
            negative,
            number_of_real_tokens=2,
            initial_basis=basis,
            number_of_steps=5,
            guidance_scale=7.5,
            eta=0.0,
            eta_seed=20800,
            rho=1.25,
            start=999,
            end=500,
            schedule_power=2.0,
            mode="disjoint",
            direction_seed=777,
            number_of_refreshes=0,
            intermediate_rank=2,
            intermediate_iterations=1,
            intermediate_seed=1515,
            transition_steps=2,
            finite_difference_epsilon=0.5,
            progress=False,
        )
        outputs.append(result)
        metadata.append(details)

    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=0.0)
    assert metadata[0] == metadata[1]
    assert metadata[0]["refresh_timesteps"] == []
    assert metadata[0]["particle_rho"] == [1.25, 1.25]


def test_block_power_and_rayleigh_ritz():
    metric = torch.diag(torch.tensor([9.0, 4.0, 1.0, 0.25]))

    def known_metric(
        latent,
        timestep,
        center_real_condition,
        direction,
        base_positive,
        number_of_real_tokens,
        finite_difference_epsilon,
    ):
        del latent, timestep, center_real_condition, base_positive
        del number_of_real_tokens, finite_difference_epsilon
        flat = direction.float().flatten(1)
        return (flat @ metric).reshape_as(direction)

    saved_metric = pullback_basis.pullback_metric_matvec
    saved_device = model.device
    pullback_basis.pullback_metric_matvec = known_metric
    model.device = torch.device("cpu")
    try:
        positive = torch.zeros((1, 2, 2))
        basis, eigenvalues = pullback_basis.compute_pullback_basis(
            latent=torch.zeros((1, 1, 2, 2)),
            timestep=torch.tensor(500),
            base_positive=positive,
            number_of_real_tokens=2,
            rank=2,
            number_of_iterations=8,
            seed=515,
            finite_difference_epsilon=0.5,
            progress_label="known metric",
        )
    finally:
        pullback_basis.pullback_metric_matvec = saved_metric
        model.device = saved_device

    torch.testing.assert_close(
        eigenvalues,
        torch.tensor([9.0, 4.0]),
        rtol=2e-4,
        atol=2e-4,
    )
    projector = basis.flatten(1).T @ basis.flatten(1)
    expected = torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(projector, expected, rtol=0.0, atol=2e-3)


def test_disjoint_assignment():
    assert directions.snake_slices(11, 4) == [
        [0, 7, 8],
        [1, 6, 9],
        [2, 5, 10],
        [3, 4],
    ]


def main():
    test_clean_ddim_known_result()
    test_cads_noise_modes_are_deterministic()
    test_adaptive_pullback_is_deterministic()
    test_block_power_and_rayleigh_ritz()
    test_disjoint_assignment()
    print("generation math tests passed")


if __name__ == "__main__":
    main()
